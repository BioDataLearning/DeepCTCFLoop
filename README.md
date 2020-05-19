# DeepCTCFLoop
DeepCTCFLoop is a deep learning model to predict whether a chromatin loop can be formed between a pair of convergent or tandem CTCF motifs using only the DNA sequences of the motifs and their flanking regions, and learn the sequence patterns hidden in the adjacent sequences of CTCF motifs. The DeepCTCFLoop model is implemented in Python using Keras 2.2.4 on a high performance computing cluster.

## Requirements
- python3
- numpy 
- pandas
- sklearn
- keras >=2.0
- tensorflow
- h5py
- scipy
- Bio

## Input Format
The input files are in FASTA format, but the description line (the line with ">" symbol in the begining) should start with class label. An example seqeunce is as follows:
```
>1
GGCTTCAGGGGAAGCCCTTCCCTGTATCCAGCCCAGTCATGACCCTTCCTGGTGGGAGGGTGGCTGTAGGATGAGGAATAGTCAGGGCCCCCCTGCACTTCAGGCAGCAGCACCCCTCAGTCCAGGGAGGAGGAATAACCCAGTTCTACGGTGGAGGCAGAGACCAGACCCGGGCCTGGGGGGCAAGTCGGGGGGCGGGGGGAGGTCGGGCAGGGTCCCCTGGGAGGATGGGGACGTGCTGTGCCCCTAGCGGCCACCAGAGGGCACCAGGACACCACTGCGGTCGGCTCAGCGGCTCCTGCCCTGGTCAGGGGGCGCCAGGTCCTGCCCCTCCTGGGGAGGGCGGGGGGCGAGAAGGGCGATTCTGGGGGCGGTTGCTCGGCTCCCTTTCCTGAAGCCATGTGGCCCGGGTGAAAGTCCCGGACCTTGAGACAGTCGGGGAGCTGATCCATGAGCAACCATCGCATGCCGGGTACAGCCGCAAACAATGTACACAGCTGCGGAATTATTTTTCtttcttcccagcacgttgggaggccgaggcaggagcatcgtctcgggccaggagttcgagaccagcctgggcaacatagggagatccccatctctacaaaaaaaaaataaataaaaaCGATCCTCAGGCTCCTCCTTCCATCCCGAACTCCCTGCCTGTCTAGGGACTGGGGGAGGCCCAGAGCCCCCCCAAATTTCTTCCCAAGCCGGGGCCCAGGGGCTGGGGAGATTCAGGCCGGCAGCTCCCCTGGGATGACCCCCAGCAGGAGGCGGCGGCGACTGCGGCCACTGGGGGCGCTCGCGGGCTTTCCCGCCCGGGCCCAGCGCTCAGGCCGGGGAGGGACGCGGGACGGGGACACCATCCACACCACCTCCTGGGCTACGGGGCCGGTCCGAGCCTCTCCAGCGCCGAGGCGCGCGCAGAGAGGTCGGGGCAGGAGAGCCGGGGCTTAAGTGCTCGCGGGGCCACCTCCGAGACGCCCGCCCAGCCCCCGCGGCTTTCCCCAGTCGGCGAGGGGCGCGGAC
```
## Training and Evaluation
The DeepCTCFLoop was evaluated on three different cell types GM12878, Hela and K562. The input data from each cell type for model training and evaluation are available in the Data directory. If you want to train your own model with DeepCTCFLoop, you can just substitute the input with your own data. The command line to train and evaluate DeepCTCFLoop as follows:
```
$ python train.py -f GM12878_pos_seq.fasta -n GM12878_neg_seq.fasta -o GM12878.output
```
During the training, the best weights will be automatically stored in a "hdf5" file. One fully trained model using the data from GM12878 has been uploaded in the Data directory.

## Motif Visualization
If you want to visualize the kernals of the first convolution layer and get its frequency and location information, you can run the following command line：
```
$ python get_motifs.py -f GM12878_pos_seq.fasta -n GM12878_neg_seq.fasta
```
Please make sure to download the "hdf5" file in the Data directory or generate your own best weights.

