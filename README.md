# Polarization differential interference contrast microscopy with physics-inspired plug-and-play denoiser for single-shot high-performance quantitative phase imaging
## M. Aleksandrovych, M. Strassberg, X. Yuan, J. Melamed, and M. Xu

Single-shot high-performance quantitative phase imaging with a Physics-inspired plug-and-play denoiser for polarization differential interference contrast (PDIC) microscopy. The quantitative phase is recovered by the alternating direction method of multipliers (ADMM), balancing total variance regularization and a pre-trained Dense Residual U-net (DRUNet) denoiser. The custom DRUNet uses the Tanh activation function to guarantee the symmetry requirement for phase retrieval.

# Application
We provide the Simense star data and full code proposed in the paper
# Results
The proposed phase retrieval method was first validated using a simulated target of Siemens star with a modified version of microlith  under the identical condition of the experimental PDIC microscope using a 60x objective. 
![Alt text](results/simens_all.png "Fig. 1. Phase retrieval of simulated Siemens star. (a) Ground truth, and the reconstructed phase using (b) the Fourier Transform solution, (c) the total variance regularized solution, and (d) the TV+DRUNet solution.")

Fig. 1. Phase retrieval of simulated Siemens star. (a) Ground truth, and the reconstructed phase using (b) the Fourier Transform solution, (c) the total variance regularized solution, and (d) the TV+DRUNet solution.

![Alt text](results/40x-cancer-tissue.jpg "Fig. 2. Prostate cancer tissue (adenocarcinoma, stage III) measured with 40x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Yellow arrows point to representative regions.")

Fig. 2. Prostate cancer tissue (adenocarcinoma, stage III) measured with 40x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Yellow arrows point to representative regions.


![Alt text](results/60x-cancer-tissue.png "Fig. 3. Prostate cancer adjacent normal tissue measured with 60x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Additionally, the region marked by the yellow rectangle in the H&E-stained histopathological image was enlarged in (e)-(h). Yellow arrows point to representative regions.")

Fig. 3. Prostate cancer adjacent normal tissue measured with 60x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Additionally, the region marked by the yellow rectangle in the H&E-stained histopathological image was enlarged in (e)-(h). Yellow arrows point to representative regions.

# Citation

This repository relies on methodologies developed by Zhang Kai in the [DPIR repository](https://github.com/cszn/DPIR). We acknowledge the utility of the DRUNet (Deep Recursive Unfolding Network) and associated utility functions, collectively known as "utils," in the advancement of our image restoration research.

https://github.com/MariiaAleksandrovych/pdic-dru/blob/d50a34af3d79dbe567f06626a8bb4d8b04ff4ac5/citations/zang2021pnp.bib

@article{
	Strassberg_Shevtsova_Kamel_Wagoner-oshima_Zhong_Xu_2021, 
 	DOI={10.1101/2021.06.06.447109}, 
  	journal={Single snapshot quantitative phase imaging with polarization differential interference contrast}, 
   	author={Strassberg, Mark and Shevtsova, Yana and Kamel, Domenick and Wagoner-oshima, Kai and Zhong, Hualin and Xu, Min}, 
    	year={2021}} 	
     	journal = {bioRxiv}
}

@article{zhang2020plug,
title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint},
year={2020}
}
