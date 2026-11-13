# Scientific References

The references below support the equations, algorithms and methods used in the three notebooks. Every DOI and URL was checked to resolve before publication. Entries marked with the challenge number indicate where they are used.

## Textbooks

1. M. A. Nielsen and I. L. Chuang, *Quantum Computation and Quantum Information*, 10th anniversary edition, Cambridge University Press (2010). Background for all challenges: Bell states and the CHSH inequality (Sections 2.6 and 12.6), Grover search (Chapter 6), quantum noise and quantum operations (Chapter 8).
2. A. Szabo and N. S. Ostlund, *Modern Quantum Chemistry: Introduction to Advanced Electronic Structure Theory*, Dover (1996). Hartree-Fock theory, configuration interaction and molecular integrals (Challenge 3).
3. T. Helgaker, P. Jorgensen and J. Olsen, *Molecular Electronic-Structure Theory*, Wiley (2000). Second quantization and the electronic Hamiltonian in chemists' notation (Challenge 3).

## Challenge 1: E91 quantum key distribution

1. A. K. Ekert, "Quantum cryptography based on Bell's theorem," *Physical Review Letters* 67, 661 (1991). DOI: 10.1103/PhysRevLett.67.661. The E91 protocol and its measurement settings.
2. J. S. Bell, "On the Einstein Podolsky Rosen paradox," *Physics Physique Fizika* 1, 195 (1964). DOI: 10.1103/PhysicsPhysiqueFizika.1.195. Bell's theorem.
3. J. F. Clauser, M. A. Horne, A. Shimony and R. A. Holt, "Proposed experiment to test local hidden-variable theories," *Physical Review Letters* 23, 880 (1969). DOI: 10.1103/PhysRevLett.23.880. The CHSH inequality $S \le 2$.
4. B. S. Cirel'son, "Quantum generalizations of Bell's inequality," *Letters in Mathematical Physics* 4, 93 (1980). DOI: 10.1007/BF00417500. The quantum bound $S \le 2\sqrt{2}$.
5. A. Acin, N. Brunner, N. Gisin, S. Massar, S. Pironio and V. Scarani, "Device-independent security of quantum cryptography against collective attacks," *Physical Review Letters* 98, 230501 (2007). DOI: 10.1103/PhysRevLett.98.230501. Key rates derived from the CHSH value.
6. V. Scarani, H. Bechmann-Pasquinucci, N. J. Cerf, M. Dusek, N. Lutkenhaus and M. Peev, "The security of practical quantum key distribution," *Reviews of Modern Physics* 81, 1301 (2009). DOI: 10.1103/RevModPhys.81.1301. Practical security, finite-size effects, error correction and privacy amplification.

## Challenge 2: Grover search under noise

1. L. K. Grover, "A fast quantum mechanical algorithm for database search," in *Proceedings of the 28th Annual ACM Symposium on Theory of Computing*, 212 (1996). DOI: 10.1145/237814.237866.
2. L. K. Grover, "Quantum mechanics helps in searching for a needle in a haystack," *Physical Review Letters* 79, 325 (1997). DOI: 10.1103/PhysRevLett.79.325.
3. C. H. Bennett, E. Bernstein, G. Brassard and U. Vazirani, "Strengths and weaknesses of quantum computing," *SIAM Journal on Computing* 26, 1510 (1997). DOI: 10.1137/S0097539796300933. Lower bound for unstructured search.
4. M. Boyer, G. Brassard, P. Hoyer and A. Tapp, "Tight bounds on quantum searching," *Fortschritte der Physik* 46, 493 (1998). arXiv: quant-ph/9605034. Success probability $\sin^2((2r+1)\theta)$ and the case of several marked items.
5. C. Zalka, "Grover's quantum searching algorithm is optimal," *Physical Review A* 60, 2746 (1999). DOI: 10.1103/PhysRevA.60.2746.
6. A. Barenco, C. H. Bennett, R. Cleve, D. P. DiVincenzo, N. Margolus, P. Shor, T. Sleator, J. A. Smolin and H. Weinfurter, "Elementary gates for quantum computation," *Physical Review A* 52, 3457 (1995). DOI: 10.1103/PhysRevA.52.3457. Decomposition of multi-controlled gates.

## Challenge 3: Variational quantum eigensolver

1. A. Peruzzo, J. McClean, P. Shadbolt, M.-H. Yung, X.-Q. Zhou, P. J. Love, A. Aspuru-Guzik and J. L. O'Brien, "A variational eigenvalue solver on a photonic quantum processor," *Nature Communications* 5, 4213 (2014). DOI: 10.1038/ncomms5213.
2. J. R. McClean, J. Romero, R. Babbush and A. Aspuru-Guzik, "The theory of variational hybrid quantum-classical algorithms," *New Journal of Physics* 18, 023023 (2016). DOI: 10.1088/1367-2630/18/2/023023.
3. A. Kandala, A. Mezzacapo, K. Temme, M. Takita, M. Brink, J. M. Chow and J. M. Gambetta, "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets," *Nature* 549, 242 (2017). DOI: 10.1038/nature23879.
4. P. Jordan and E. Wigner, "Uber das Paulische Aquivalenzverbot," *Zeitschrift fur Physik* 47, 631 (1928). DOI: 10.1007/BF01331938. The Jordan-Wigner transformation.
5. S. Bravyi, J. M. Gambetta, A. Mezzacapo and K. Temme, "Tapering off qubits to simulate fermionic Hamiltonians," arXiv: 1701.08213 (2017). Symmetry-based qubit reduction.
6. V. Verteletskyi, T.-C. Yen and A. F. Izmaylov, "Measurement optimization in the variational quantum eigensolver using a minimum clique cover," *Journal of Chemical Physics* 152, 124114 (2020). DOI: 10.1063/1.5141458. Grouping of commuting Pauli terms.
7. K. M. Nakanishi, K. Fujii and S. Todo, "Sequential minimal optimization for quantum-classical hybrid algorithms," *Physical Review Research* 2, 043158 (2020). DOI: 10.1103/PhysRevResearch.2.043158. Coordinate-wise minimization of sinusoidal energy landscapes.
8. M. Ostaszewski, E. Grant and M. Benedetti, "Structure optimization for parameterized quantum circuits," *Quantum* 5, 391 (2021). DOI: 10.22331/q-2021-01-28-391. Rotosolve.
9. J. C. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation," *IEEE Transactions on Automatic Control* 37, 332 (1992). DOI: 10.1109/9.119632. The SPSA optimizer.
10. K. Temme, S. Bravyi and J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Physical Review Letters* 119, 180509 (2017). DOI: 10.1103/PhysRevLett.119.180509. Zero-noise extrapolation.
11. A. Kandala, K. Temme, A. D. Corcoles, A. Mezzacapo, J. M. Chow and J. M. Gambetta, "Error mitigation extends the computational reach of a noisy quantum processor," *Nature* 567, 491 (2019). DOI: 10.1038/s41586-019-1040-7.
12. Q. Sun et al., "PySCF: the Python-based simulations of chemistry framework," *WIREs Computational Molecular Science* 8, e1340 (2018). DOI: 10.1002/wcms.1340. Used by the organizers to compute the molecular integrals and reference energies.

## Software documentation

1. Qiskit API documentation: https://quantum.cloud.ibm.com/docs/en/api/qiskit
2. Qiskit Aer documentation: https://qiskit.github.io/qiskit-aer/
3. Qiskit Aer noise models: https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html
4. PySCF documentation: https://pyscf.org/

## Conventions used in the notebooks

* Bit strings follow Qiskit's ordering: the last character refers to qubit 0.
* In Challenge 1, the measured bit 0 corresponds to the eigenvalue $+1$ and bit 1 to $-1$.
* In Challenge 3, energies are in hartree and errors in millihartree; chemical accuracy is $1.6 \times 10^{-3}$ hartree (about 1 kcal/mol).
