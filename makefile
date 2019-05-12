TsetlinMachinePOC: 
	nvcc -o TsetlinMachinePOC main.cu tsetlin_random_wheel.cu cpu_kernel.cu gpu_kernel.cu kernels.cu

clean:
	rm TsetlinMachinePOC
