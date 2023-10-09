# Polarization differential interference contrast microscopy with physics-inspired plug-and-play denoiser for single-shot high-performance quantitative phase imaging
## M. Aleksandrovych, M. Strassberg, X. Yuan, J. Melamed, and M. Xu

Single-shot high-performance quantitative phase imaging with a Physics-inspired plug-and-play denoiser for polarization differential interference contrast (PDIC) microscopy. The quantitative phase is recovered by the alternating direction method of multipliers (ADMM), balancing total variance regularization and a pre-trained Dense Residual U-net (DRUNet) denoiser. The custom DRUNet uses the Tanh activation function to guarantee the symmetry requirement for phase retrieval.

# Results
The proposed phase retrieval method was first validated using a simulated target of Siemens star with a modified version of microlith  under the identical condition of the experimental PDIC microscope using a 60x objective. 
![Alt text](results/simens_all.png "Fig. 1. Phase retrieval of simulated Siemens star. (a) Ground truth, and the reconstructed phase using (b) the Fourier Transform solution, (c) the total variance regularized solution, and (d) the TV+DRUNet solution.")

Fig. 1. Phase retrieval of simulated Siemens star. (a) Ground truth, and the reconstructed phase using (b) the Fourier Transform solution, (c) the total variance regularized solution, and (d) the TV+DRUNet solution.

![Alt text](results/40x-cancer-tissue.jpg "Fig. 2. Prostate cancer tissue (adenocarcinoma, stage III) measured with 40x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Yellow arrows point to representative regions.")

Fig. 2. Prostate cancer tissue (adenocarcinoma, stage III) measured with 40x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Yellow arrows point to representative regions.


![Alt text](results/60x-cancer-tissue.png "Fig. 3. Prostate cancer adjacent normal tissue measured with 60x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Additionally, the region marked by the yellow rectangle in the H&E-stained histopathological image was enlarged in (e)-(h). Yellow arrows point to representative regions.")

Fig. 3. Prostate cancer adjacent normal tissue measured with 60x objective. (a) H&E-stained histopathological image and reconstructed phase images using (b) Fourier transform, (c) total variance regularized, and (d) total variance plus DRUNet denoiser methods. Additionally, the region marked by the yellow rectangle in the H&E-stained histopathological image was enlarged in (e)-(h). Yellow arrows point to representative regions.

# Citation
@article {Strassberg2021.06.06.447109,
	author = {Mark Strassberg and Yana Shevtsova and Domenick Kamel and Kai Wagoner-oshima and Hualin Zhong and Min Xu},
	title = {Single snapshot quantitative phase imaging with polarization differential interference contrast},
	elocation-id = {2021.06.06.447109},
	year = {2021},
	doi = {10.1101/2021.06.06.447109},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {We present quantitative phase imaging with polarization differential interference contrast (PDIC) realized on a slightly modified differential interference contrast (DIC) microscope. By recording the Stokes vector rather than the intensity of the differential interference pattern with a polarization camera, PDIC enables single snapshot quantitative phase imaging with high spatial resolution in real-time at speed limited by the camera frame rate alone. The approach applies to either absorptive or transparent samples and can integrate simply with fluorescence imaging for co-registered simultaneous measurements. Furthermore, an algorithm with total variation regularization is introduced to solve the quantitative phase map from partial derivatives. After quantifying the accuracy of PDIC phase imaging with numerical simulations and phantom measurements, we demonstrate the biomedical applications by imaging the quantitative phase of both stained and unstained histological tissue sections and visualizing the fission yeast Schizosaccharomyces pombe{\textquoteright}s cytokinesis.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2021/06/06/2021.06.06.447109},
	eprint = {https://www.biorxiv.org/content/early/2021/06/06/2021.06.06.447109.full.pdf},
	journal = {bioRxiv}
}
