import os
import jax
import jax.numpy as jnp
# Removed flax, flax.linen imports
import optax
import numpy as np
import gc
from tqdm import tqdm
import time
import math
import logging
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from functools import partial
# from flax.training import train_state
from flax import struct # Keep for TrainState definition initially
import jmp  # JMP 라이브러리 추가
import functools
from torch.utils import data
from data import LibriTTSDataset, MultipliedDataset  # 기존 데이터셋 클래스 import
from multiprocessing import cpu_count
# import orbax.checkpoint as ocp
from flax import nnx # Import NNX
from codec_module import CodecModule
from data import NumpyLoader
from scalax.sharding import PartitionSpec, MeshShardingHelper, FSDPShardingRule
from flax.training import checkpoints

# 학습률 스케줄 함수 생성 (No change needed)
def create_learning_rate_fn(base_learning_rate, warmup_steps, decay_steps, end_learning_rate=1e-7):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=warmup_steps
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=decay_steps,
        alpha=end_learning_rate / base_learning_rate
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )
    return schedule_fn

class AdvTrainState(struct.PyTreeNode): # Keep PyTreeNode for JAX compatibility
    step: int
    # Store the static graph definition
    graphdef: nnx.GraphDef #= struct.field(pytree_node=False) # Don't treat GraphDef as a leaf
    # Store the dynamic state parts (these are PyTrees)
    gen_params: nnx.State # Holds Generator params + state
    disc_params: nnx.State # Holds Discriminator params + state
    gen_opt_state: optax.OptState
    disc_opt_state: optax.OptState
    other_states: nnx.State # Potentially hold BatchNorm state etc. if split out
    # We don't use create classmethod here as GraphDef needs special handling

# GAN 학습을 위한 함수 추가
def train_adv(
        index,
        # model argument removed, we instantiate it inside
        mp_policy,
        train_loader,
        train_steps=1000,
        per_device_batch=16,
        sample_rate=24000,
        val_freq=1,
        visualize_freq=1,
        epochs=5,
        seed=42,
        weight_decay=0.01,
        gen_lr=2e-4,
        disc_lr=2e-4,
        lambda_mel_loss=15.0, # Pass these through, Model now reads them internally
        lambda_adv=1.0,
        lambda_feat_match=1.0,
        lambda_disc=1.0,
        bf16=False, # bf16 handled by JMP policy
        gen_grad_clip=1.0,
        disc_grad_clip=1.0,
        warmup_steps=1000,
        decay_steps=10000,
        end_lr_ratio=1e-2
        ):

    # 랜덤 시드 설정
    key = jax.random.PRNGKey(seed)
    model_key, train_key = jax.random.split(key) # Split key for model init and training

    # Define a 1D mesh with a `fsdp` axis only
    mesh = MeshShardingHelper([-1], ['fsdp'])
    # Use FSDP sharding rule with axis name `fsdp`
    model_sharding_rule = FSDPShardingRule(fsdp_axis_name='fsdp')
    # Initialize the NNX Model directly
    # 학습률 스케줄 함수 생성 (No change)
    # 옵티마이저 설정 (No change in definition)
    # checkpointer = ocp.StandardCheckpointer()
    gen_lr_schedule = create_learning_rate_fn(gen_lr, warmup_steps, decay_steps, gen_lr * end_lr_ratio)
    disc_lr_schedule = create_learning_rate_fn(disc_lr, warmup_steps, decay_steps, disc_lr * end_lr_ratio)
    gen_tx = optax.chain(
        optax.clip_by_global_norm(gen_grad_clip),
        optax.adamw(gen_lr_schedule, b1=0.9, b2=0.999, weight_decay=weight_decay)
    )
    disc_tx = optax.chain(
        optax.clip_by_global_norm(disc_grad_clip),
        optax.adamw(disc_lr_schedule, b1=0.9, b2=0.999, weight_decay=weight_decay)
    )
    @partial(
        mesh.sjit,  # Shard the initialization function
        out_shardings=model_sharding_rule,  # The initialized model and optimizer states should be sharded by FSDP
    )
    def init_fn(rng):
        model = CodecModule(rngs=nnx.Rngs(rng))
        # Split the initial model state into graph definition and parameters/state
        # We need to filter params for generator and discriminator separately for optimizers.
        # Using the path filtering approach defined above:
        graphdef, params, other_states = nnx.split(model, nnx.Param, ...)
        params = params.to_pure_dict()
        gen_params = nnx.State({'CodecEnc': params['CodecEnc'], 'generator': params['generator']}, _copy=False)
        disc_params = nnx.State({'discriminator': params['discriminator'], 'spec_discriminator': params['spec_discriminator']}, _copy=False)
        # Combine gen/disc non-param state with rest_state if needed, or handle separately.
        # For simplicity now, assume optimizers only need the filtered params.
        # breakpoint()
        # Initialize the state
        return AdvTrainState(
            step=jnp.array(0),
            graphdef=graphdef,
            gen_params=gen_params, # Pass the filtered state trees
            disc_params=disc_params,
            other_states=other_states,
            gen_opt_state=gen_tx.init(gen_params), # Initialize optimizer with filtered params
            disc_opt_state=disc_tx.init(disc_params),
        )

    initial_state_obj = init_fn(train_key)

    if jax.process_index() == 0:
        # dummy_input for printing info
        dummy_input = jnp.ones((per_device_batch, 16000, 1), dtype=jnp.float32)
        print(f"입력 데이터 형식: {dummy_input.shape}, 타입: {dummy_input.dtype}")
        print(f"JMP 정책: params={mp_policy.param_dtype}, compute={mp_policy.compute_dtype}, output={mp_policy.output_dtype}")
        print(f"생성자 기본 학습률: {gen_lr}, 판별자 기본 학습률: {disc_lr}")
        print(f"워밍업 스텝: {warmup_steps}, 감쇠 스텝: {decay_steps}")

    # devices = mesh_utils.create_device_mesh((jax.local_device_count(),))
    # print(f'devices: {devices}')
    # print(f'devices_shape: {devices.shape}')
    # mesh = Mesh(devices, axis_names=('batch',))
    # data_sharding = NamedSharding(mesh, P(('batch',)))
    # state_sharding = NamedSharding(mesh, P())

    # sharded_state = initial_state_obj #jax.device_put(initial_state_obj, state_sharding)

    # 판별자 손실 함수 - Needs GraphDef to reconstruct the model
    def disc_loss_fn(disc_params, state, inputs):
        # NNX state update approach: merge -> call -> split
        # Merge the *current* state parts with the static graphdef
        # Combine all relevant state parts before merging
        # Assuming rest_state is needed and implicitly handled or not updated in loss path
        gen_params, disc_params, inputs = mp_policy.cast_to_compute((state.gen_params, disc_params, inputs))
        current_state = nnx.merge_state(gen_params, disc_params, state.other_states) # , rest_state if needed

        # Reconstruct the model inside the function
        model = nnx.merge(state.graphdef, current_state)

        # Cast inputs only, model state types handled by NNX/JMP interaction
        # inputs = mp_policy.cast_to_compute(inputs)
        # breakpoint()

        # Perform forward pass and loss calculation using the merged model
        outputs = model.forward(inputs)
        disc_outputs = model.compute_disc_loss(outputs) # Call method directly
        disc_loss = disc_outputs['disc_loss']

        # Get potentially updated state *if* loss calculation modifies state (e.g., BatchNorm)
        # _, updated_state = nnx.split(model) # Split again to capture updates
        # We might need to filter updated_state and return it if non-param state changes.
        # For now, assume only loss and grads are needed.

        current_lr = disc_lr_schedule(state.step)
        metrics = {
            'real_loss': mp_policy.cast_to_output(disc_outputs['real_loss']),
            'fake_loss': mp_policy.cast_to_output(disc_outputs['fake_loss']),
            'disc_loss': mp_policy.cast_to_output(disc_loss),
            'disc_lr': mp_policy.cast_to_output(current_lr)
        }
        return mp_policy.cast_to_output(disc_loss), metrics # Return only metrics for now

    # 생성자 손실 함수 - Needs GraphDef
    def gen_loss_fn(gen_params, state, inputs):
        # Merge the *current* state parts
        gen_params, disc_params, inputs = mp_policy.cast_to_compute((gen_params, state.disc_params, inputs))
        current_state = nnx.merge_state(gen_params, disc_params, state.other_states) # , rest_state if needed
        model = nnx.merge(state.graphdef, current_state)

        # inputs = mp_policy.cast_to_compute(inputs)

        # Run forward and generator loss calculation
        outputs = model.forward(inputs)
        # compute_gen_loss now returns a dictionary of losses
        gen_outputs = model.compute_gen_loss(outputs)
        gen_loss = gen_outputs['gen_loss']

        # Potentially split and return updated state if forward/gen_loss modify it.
        # _, updated_state = nnx.split(model)

        current_lr = gen_lr_schedule(state.step)
        # Include all computed losses in metrics
        metrics = {k: mp_policy.cast_to_output(v) for k, v in gen_outputs.items()}
        metrics['gen_lr'] = mp_policy.cast_to_output(current_lr)

        return mp_policy.cast_to_output(gen_loss), metrics # Return only metrics

    # 학습 스텝 함수 - Adapt to use AdvTrainState and pass GraphDef
    disc_grad_fn = jax.value_and_grad(disc_loss_fn, has_aux=True) # Grad w.r.t disc_params_state
    gen_grad_fn = jax.value_and_grad(gen_loss_fn, has_aux=True) # Grad w.r.t gen_params_state

    # @functools.partial(jax.pmap, axis_name='batch')
    # Define the jitted train step function
    # @partial(jax.jit,
    #          in_shardings=(state_sharding, data_sharding), # Sharding for state and inputs
    #          out_shardings=(state_sharding, None), # Output sharding for state and metrics
    #          static_argnums=() # No static args for now
    #         )
    @partial(
        mesh.sjit,  # Shard the train step function
        # The input model should be sharded with FSDP, the data is loaded as replicated
        in_shardings=(model_sharding_rule, PartitionSpec()),
        # The output model should be sharded with FSDP, the loss metrics is replicated
        out_shardings=(model_sharding_rule, PartitionSpec()),
        # After the beginning of the function, the data should be sharded along the `fsdp` axis
        args_sharding_constraint=(model_sharding_rule, PartitionSpec('fsdp')),
        donate_argnums=(0,),  # No need to preserve the old model
    )
    def train_step(state, inputs):
        # 1. 판별자 업데이트
        # Pass graphdef from state object
        (disc_loss, disc_metrics), disc_grads = disc_grad_fn(
            state.disc_params, state, inputs
        )

        # disc_grads = jax.lax.pmean(disc_grads, axis_name='batch')
        # Apply updates to the discriminator state tree
        updates, new_disc_opt_state = disc_tx.update(disc_grads, state.disc_opt_state, state.disc_params)
        new_disc_params = optax.apply_updates(state.disc_params, updates)

        state = state.replace(
            disc_params=new_disc_params,
            disc_opt_state=new_disc_opt_state
        )
        # 2. 생성자 업데이트
        # Pass the *original* disc_params state to gen_loss_fn, as disc wasn't updated for gen's forward pass view
        (gen_loss, gen_metrics), gen_grads = gen_grad_fn(
            state.gen_params, state, inputs
        )
        # gen_grads = jax.lax.pmean(gen_grads, axis_name='batch')
        # Apply updates to the generator state tree
        updates, new_gen_opt_state = gen_tx.update(gen_grads, state.gen_opt_state, state.gen_params)
        new_gen_params = optax.apply_updates(state.gen_params, updates)
        # breakpoint()
        state = state.replace(
            gen_params=new_gen_params,
            gen_opt_state=new_gen_opt_state
        )
        # 메트릭 병합
        combined_metrics = {}
        combined_metrics.update(disc_metrics)
        combined_metrics.update(gen_metrics)

        # Return the updated state object
        # Note: If loss functions returned updated non-param state, merge it back here.
        return state.replace(
            step=state.step + 1,
            # gen_params=new_gen_params,
            # disc_params=new_disc_params,
            # gen_opt_state=new_gen_opt_state,
            # disc_opt_state=new_disc_opt_state,
            # other_state=updated_other_state, # Update if managed
        ), combined_metrics

    # 디바이스 정보 가져오기 (No change)
    devices = jax.devices()
    device_count = len(devices)

    # 훈련 상태를 모든 장치에 복제
    # JAX automatically handles PyTree replication including the AdvTrainState
    # state_replicated = jax.device_put_replicated(initial_state_obj, devices)

    # 성능 측정을 위한 변수들 (No change)
    start_time = time.time()
    last_time = start_time
    total_steps = 0

    # 오디오 전처리 및 형식 변환 함수 (No change needed)
    def preprocess_audio(waveform_batch):
        audio_data = waveform_batch[0]
        # Transpose for (batch, time, channel) - Check if NNX modules expect this or (B, C, T)
        # Assuming the NNX modules were adapted to expect (B, T, C) based on the transpose here.
        # If they expect (B, C, T), remove the transpose. Let's keep it for now.
        return np.transpose(audio_data, (0, 2, 1))

    # # 데이터 준비 함수 - Adapt input casting if needed
    # def prepare_batch_for_tpu(batch):
    #     audio_data = preprocess_audio(batch) # Shape (B, T, C) potentially
    #     per_device_batch = audio_data.shape[0] // device_count
    #     audio_data = audio_data[:per_device_batch * device_count]
    #     audio_data = audio_data.reshape(device_count, per_device_batch, *audio_data.shape[1:])

    #     # Cast inputs to compute dtype. Model params are handled by JMP policy interaction
    #     # within the train step where model is merged/called.
    #     audio_data = mp_policy.cast_to_compute(audio_data)
    #     return audio_data
    # --- Data Preparation (Sharding Aware) ---

    def prepare_global_batch(batch):
        """Prepares a global batch, does not shard yet."""
        audio_data = preprocess_audio(batch) # (GlobalB, T, C)
        # Ensure the global batch size is divisible by device count
        # This should be handled by DataLoader drop_last=True if batch_size is multiple of device_count
        # Add check just in case
        # num_devices = mesh.size
        # if audio_data.shape[0] % num_devices != 0:
        #      raise ValueError(f"Global batch size {audio_data.shape[0]} not divisible by device count {num_devices}")
        # # No need to reshape here, just return the global batch
        return jnp.asarray(audio_data) # Convert to JAX array

    # 훈련 루프 (Logic remains the same, uses new state)
    current_state = initial_state_obj
    for epoch in range(1, epochs+1):
        step = 0
        for batch in train_loader:
            # inputs = prepare_batch_for_tpu(batch)
            inputs = prepare_global_batch(batch)
            # sharded_batch = jax.device_put(inputs, data_sharding)
            # inputs = jnp.ones((device_count, per_device_batch, 16000, 1), dtype=jnp.float32)
            current_state, metrics = train_step(current_state, inputs)

            total_steps += 1
            step += 1
            if total_steps >= train_steps: # Check if train_steps is used
                break

            if step % 10 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                steps_per_sec = 10 / elapsed if elapsed > 0 else 0
                avg_steps_per_sec = total_steps / (current_time - start_time) if (current_time - start_time) > 0 else 0

                metrics = jax.device_get(metrics)
                if jax.process_index() == 0:
                    loss_str = ", ".join([f"{k}={float(v):.4f}" for k, v in metrics.items() if 'loss' in k])
                    lr_str = f"gen_lr={float(metrics['gen_lr']):.1E}, disc_lr={float(metrics['disc_lr']):.1E}"
                    print(f'Epoch {epoch}, Step {step}, Losses: {loss_str}, LRs: {lr_str}, Steps/sec (curr): {float(steps_per_sec):.2f}, Steps/sec (avg): {float(avg_steps_per_sec):.2f}')

                last_time = current_time

        checkpoints.save_checkpoint_multiprocess('/home/hoyeol/Audio_Tokenization/BigCodec_NNX/bigcodec_nnx_checkpoints', current_state, step)
        
        if total_steps >= train_steps:
             break # Break outer loop too

    # 전체 학습 성능 통계 (No change)
    total_time = time.time() - start_time
    avg_steps_per_sec = total_steps / total_time if total_time > 0 else 0
    if jax.process_index() == 0:
        print('\n훈련 완료:\n총 스텝: {}\n총 소요 시간: {:.2f}초\n평균 스텝/초: {:.2f}'.format(
            total_steps, float(total_time), float(avg_steps_per_sec)))

    return current_state


def main():
    # 데이터셋 로드 (No change)
    train_dataset = LibriTTSDataset(
        root="/mnt/disks/persist/data/LibriTTS",
        subsets=["train-clean-100", "train-clean-360", "train-other-500"],
        sample_rate=16000, # Note: sample rate mismatch with train_adv default (24k)? Using 16k.
        duration=1.0,
        offset_mode="random",
    )

    # 데이터 로더 생성 (No change)
    device_count = jax.local_device_count() # Use local_device_count
    per_device_batch = 16
    total_batch_size = per_device_batch * device_count

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=total_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min(cpu_count(), 60), # Use min to avoid oversubscribing
        drop_last=True,
    )

    if jax.process_index() == 0:
        print(f"Local device count: {device_count}")
        print(f"데이터셋 크기: {len(train_dataset)}")
        print(f"배치 크기: {total_batch_size} (디바이스당 {per_device_batch})")

    # 모델 초기화는 train_adv 안에서 수행됨

    # JMP 정책 생성 (No change)
    mp_policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    train_steps_limit = 300000 # Max steps from original code
    # 총 학습 스텝 계산 (Logic Adjusted slightly)
    steps_per_epoch = len(train_dataset) // total_batch_size
    epochs = train_steps_limit // steps_per_epoch# 60 // MULT
    print(f"Epochs: {epochs}")

    # 학습률 스케줄링 파라미터 설정 (Adjust decay steps based on estimate or fallback)
    warmup_steps = 1000 # Fixed warmup
    decay_steps = 500000

    # train_adv 호출 - model 객체는 내부에서 생성되므로 전달 안 함
    final_state_host = train_adv(
        index=0, # Assuming index is for compatibility, not used internally
        mp_policy=mp_policy,
        train_loader=train_loader,
        train_steps=train_steps_limit, # Pass the step limit
        epochs=epochs,
        per_device_batch=per_device_batch,
        sample_rate=16000, # Match dataset
        gen_lr=1e-4,
        disc_lr=1e-4,
        lambda_mel_loss=15.0,
        lambda_adv=1.0,
        lambda_feat_match=1.0,
        lambda_disc=1.0,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_lr_ratio=1e-1,
        gen_grad_clip=1.0, # Added missing args
        disc_grad_clip=1.0 # Added missing args
    )


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()