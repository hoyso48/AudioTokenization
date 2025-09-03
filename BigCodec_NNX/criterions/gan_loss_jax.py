import jax
import jax.numpy as jnp
from flax import nnx

class GANLoss(nnx.Module):
    """
    JAX 기반 GAN 손실 함수 클래스
    PyTorch 버전과 유사한 인터페이스를 제공합니다.
    """
    def disc_loss(self, real, fake):
        """
        판별자(Discriminator) 손실 함수
        
        Args:
            real: 실제 데이터에 대한 판별자 출력
            fake: 생성된 데이터에 대한 판별자 출력
        
        Returns:
            real_loss: 실제 데이터에 대한 MSE 손실
            fake_loss: 가짜 데이터에 대한 MSE 손실
        """
        real_loss = jnp.mean((real - 1.0) ** 2)  # 실제 데이터는 1에 가까워야 함          
        fake_loss = jnp.mean(fake ** 2)  # 가짜 데이터는 0에 가까워야 함
        return real_loss, fake_loss
    
    def gen_loss(self, fake):
        """
        생성자(Generator) 손실 함수     
        Args:
            fake: 생성된 데이터에 대한 판별자 출력
        
        Returns:
            gen_loss: 생성자 MSE 손실
        """
        gen_loss = jnp.mean((fake - 1.0) ** 2)  # 가짜 데이터를 실제처럼 (1) 보이게 하는 손실
        return gen_loss