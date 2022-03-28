import numpy as np 
import matplotlib.pyplot as plt
from ..utils.misc import add_noise
from .library_tools import delete_term
from .anise import Anise
from .pde import PDE 
from pathlib import Path

parent = Path(__file__).parent.parent

style_file = (parent / "prx.mplstyle").resolve()
plt.style.use(str(style_file))

class Benchmark(Anise):

    def add_noise_all(self, noise_strength=0.01, seed=None):
        '''
        add_noise_all(noise_strength, seed=None)

        Function to add white Gaussian noise to all the fields in place. (Check the method `reset_data` to reset the data.) Uses NumPy's  random.default_rng() generator if available, and random.randn if not. This is done using a helper function `add_noise` (imported from the `utils`) to add the noise to each individual field. 

        Parameters
        ----------
        noise_strength : float
            Strength of the noise relative to the standard deviation. This is done per field across space and time. 
            Default is 0.01
        seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
            Seed for NumPy's default_rng(). From its description:
            A seed to initialize the `BitGenerator`. If None, then fresh, unpredictable entropy will be pulled from the OS. If an ``int`` or ``array_like[ints]`` is passed, then it will be passed to `SeedSequence` to derive the initial `BitGenerator` state. One may also pass in a`SeedSequence` instance.
            Additionally, when passed a `BitGenerator`, it will be wrapped by `Generator`. If passed a `Generator`, it will be returned unaltered.
            If default_rng is not available, this will be used as a seed for the random.randn
        '''
        self.Qxx_all = add_noise(self.Qxx_all, noise_strength, seed)
        self.Qxy_all = add_noise(self.Qxy_all, noise_strength, seed)
        self.u_all = add_noise(self.u_all, noise_strength, seed)
        self.v_all = add_noise(self.v_all, noise_strength, seed)

    def stokes_int(self):

        print("Generating libraries...")
        (lib_lhs, lib_Q, lib_NS, lib_Stokes,
         lib_overdamped) = self.generate_libraries_int()
        
        print("Computing the PDE for Stokes flow...")
        pde_St = PDE(lib_Stokes, lib_lhs, '∇²ω', self.metadata)
        print("Done! Stored under pde_St.\n")

        ida1 = np.argwhere(
            pde_St.rhs["name"] == "(∂²Qxy/∂x²)").flatten()[0]
        ida2 = np.argwhere(
            pde_St.rhs["name"] == "(∂²Qxx/∂x∂y)").flatten()[0]
        ida3 = np.argwhere(
            pde_St.rhs["name"] == "(∂²Qxy/∂y²)").flatten()[0]

        w = pde_St.w_all[-4]

        alpha = np.mean([w[ida1],-0.5*w[ida2], -w[ida3]])

        return alpha

    def stokes_weak(self):

        (lib_lhs, lib_NS, lib_St) = self.weak_form_flow_libs(1,
                                                             500,
                                                             (45,45,101))

        lib_St = delete_term(lib_St, "∇⁴u")

        pde_St_w = PDE(lib_St, lib_lhs, '∇²u', self.metadata)

        id = np.argwhere(pde_St_w.rhs["name"] == "∇·Q").flatten()[0]
        
        return pde_St_w.w_all[-2,id]
    
    def weak_form_benchmark_window_size(self, noise_strength=1e-1):

        samples = 3
        num_sizes = 5

        # WXs = 2*((np.linspace(4, 3*self.NX//4, num_sizes)/2).astype(int))+1
        # WYs = 2*((np.linspace(4, 3*self.NY//4, num_sizes)/2).astype(int))+1
        WTs = 2*((np.linspace(4, self.NT//8, num_sizes)/2).astype(int))+1
        WXs = 205*np.ones(num_sizes).astype(int)
        WYs = 205*np.ones(num_sizes).astype(int)
        # WTs = 45*np.ones(len(WXs)).astype(int)
        print(f"WXs = {WXs}")
        print(f"WYs = {WYs}")
        print(f"WTs = {WTs}")

        r2s = np.zeros([samples, num_sizes])
        alphas = np.zeros([samples, num_sizes])
        zetas = np.zeros([samples, num_sizes])
        frictions = np.zeros([samples, num_sizes])

        if noise_strength!=0:
            self.add_noise_all(noise_strength)

        for j in range(samples):
            for i in range(num_sizes):

                metadata = dict()
                num_windows = 50
                window_size = (WXs[i], WYs[i], WTs[i])
                # window_size = (51, 51, WTs[i])
                # window_size = (WXs[i], WYs[i], 51)
                sample = j+1
                print(f"Sample : {sample}")
                print(f"Window Size : {window_size}")


                (lib_lhs, lib_St) = self.weak_form_flow_libs(num_windows, window_size,sample)

                pde_St = PDE()
                pde_St.compute(lib_St, lib_lhs, '∇²u', self.metadata)

                # Get the value of activity in the optimal model

                alpha_id = pde_St.desc.index('∇·Q')

                opt = -pde_St.nopt # Model is stored in reverse order of sparsity
                r2s[j, i] = pde_St.r2[opt]
                alphas[j, i] = pde_St.w_all[opt][alpha_id]

        self.reset_data()

        a_mean = np.mean(alphas, axis=0)
        a_std = np.std(alphas, axis=0)

        r2s_mean = np.mean(r2s, axis=0)
        r2s_std = np.std(r2s, axis=0)

        np.savez(f'{self.data_dir}/weak_form_benchmark.npz', r2s=r2s,
                 alphas=alphas, frictions=frictions, Ws=np.array([WXs, WYs, WTs]))


        fig, ax = plt.subplots()
        ax.plot(np.arange(1, num_sizes+1), r2s_mean, color='m')
        ax.fill_between(np.arange(1, num_sizes+1), r2s_mean-r2s_std,
                        r2s_mean+r2s_std, alpha=0.3, color='m')
        plt.xlabel("Window size")
        plt.ylabel(r"$R^2$")
        ax.tick_params(direction='in')
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/rsquared_vs_window_size.png", dpi=300)
        plt.savefig(f"{self.data_dir}/rsquared_vs_window_size.svg", dpi=300)

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, num_sizes+1), a_mean, label=r'$\alpha$', color='g')
        ax.fill_between(np.arange(1, num_sizes+1), a_mean-a_std,
                        a_mean+a_std, alpha=0.3, color='g')
        plt.ylabel(r"$\alpha/\eta$")
        # ax.plot(np.arange(1,num_sizes+1), z_mean, label=r'$\zeta$', color='y')
        # ax.fill_between(np.arange(1,num_sizes+1), z_mean-z_std, z_mean+z_std, alpha=0.3, color='y')
        plt.xlabel("Window size")
        ax.tick_params(direction='in')
        # ax2 = ax.twinx()
        # ax2.plot(np.arange(1,num_sizes+1), g_mean, label=r'$\Gamma$', color='m')
        # ax2.fill_between(np.arange(1,num_sizes+1), g_mean-g_std, g_mean+g_std, alpha=0.3, color='m')

        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/alpha_vs_window_size.svg", dpi=300)
        plt.savefig(f"{self.data_dir}/alpha_vs_window_size.png", dpi=300)


    def run(self):

        noise_levels = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
        
        alphas_int = np.zeros(noise_levels.shape)

        alphas_weak = np.zeros(noise_levels.shape)
        
        for i, noise_strength in enumerate(noise_levels):

            self.reset_data()

            self.add_noise_all(noise_strength)

            alphas_int[i] = self.stokes_int()

            alphas_weak[i] = self.stokes_weak()

            print(
                f"Noise strength: {noise_strength}, Int: {alphas_int[i]}, Weak: {alphas_weak[i]}")
        
        return noise_levels, alphas_int, alphas_weak

