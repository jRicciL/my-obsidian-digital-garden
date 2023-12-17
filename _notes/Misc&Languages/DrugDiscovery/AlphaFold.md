---
---

# AlphaFold 1

- Developed by DeepMind
- Participation in the Critical Assessment of Techniques for Protein Structure Prediction ([[CASP]])

### Disadvantages and critics

-  Does not reveal the mechanism or rules of protein folding for the [protein folding problem](https://en.wikipedia.org/wiki/Protein_folding_problem "Protein folding problem") to be considered solved.[\[7\]](https://en.wikipedia.org/wiki/AlphaFold#cite_note-curry-7)[\[8\]](https://en.wikipedia.org/wiki/AlphaFold#cite_note-8)

## Articles

#### [No, DeepMind has not solved protein folding](http://occamstypewriter.org/scurry/2020/12/02/no-deepmind-has-not-solved-protein-folding/)
- DeepMind has made a big step forward.
- They are not yet at the point where we can say that protein 
- We are not at the point where this AI tool can be used for drug discovery.
- AI methods rely on learning the rules of protein folding from existing protein structures.
	- Folding bias -> Database


#### [AlphaFold2 @ CASTP: 'It feels like ones child has left home'](https://moalquraishi.wordpress.com/2020/12/08/alphafold2-casp14-it-feels-like-ones-child-has-left-home/amp/)
- CASTP organizers declared that the protein structure prediction problem for single protein chains to be solved.

##### The Advance
- They had achieved a median GDT\_TS of around 80
	-  GlobalDistanceTest\_TotalScore
	-  Where GDT\_Pn denotes percent of residues under distance cutoff <= nÅ
- Statements have been made that this is proteins’ “[ImageNet](https://en.wikipedia.org/wiki/ImageNet) moment”
	- -> was the first time deep learning demonstrated it can outperform conventional approaches
- AF2 did best for 88 out of 97 targets!
- CASP proteins are actually harder than usual.
	- but perhaps this year was an “easy” year
	- NO> the CASP14 organizers quantified the difficulty of this year’s targets and found them to be harder than those of the few previous CASPs, so this was a hard year!
	- AF2 achieves for Cα atoms an accuracy of <1Å 25% of the time, <1.6Å 50% of the time, and <2.5Å 75% of the time.
	- Not a physics-based approaches like molecular dynamics (MD).

##### A solution?
- What about unnatural amino acids?
- how good is good enough?
- the actual dynamic process by which proteins fold

##### The method
- DeepMind is in an exceedingly dominant position here—they will invariably get the cover of Nature or Science and may one day nab their first Nobel prize for AF2
- they really told us very little

**Out with the Potts models, in with raw MSAs**
- MSA of homologous protein
- Summary statistics out of this alignment
- Predict a distogram
	- A matrix of the probabilities of pairwise distances between $C_\beta$ atoms.
	- AF2 no longer summarizes the MSA.
	- it keeps all raw sequences and iteratively “attends” to them