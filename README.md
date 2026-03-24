# Machine-Learning-Project
Github for the group of Rohan, Lasaro, and Don.

Project Outline: 
  1. You are provided with two text files in the group project module, namely train and test.
     In these files, each record consists of 2 lines (the first line starts with “>”) as shown below:
        
        - >peptide|1
        - EITVEPVRHPKKDPSEAE
     
     The first line consists of “>peptide” followed by “|” and after that “0” or “1”. Peptides demonstrate the
     nature of the input that is presented in the next line. The number after the “|” is the label. Label “0”
     represents “toxic peptides” and label “0” represents non-toxic peptides. The following line is the
     corresponding peptide sequence.

2. You need to get protein sequences and extract relevant features that are representative of
    the protein sequences. You can also use the features that are already available. You are
    required to extract three different groups of features:


    2.1 Physicochemical-based features: These are extracted based on the physical, chemical, and
    physicochemical properties of proteins, peptides, and amino acids. You can get more information on how
    to extract these features from the following articles. You need to use the numerical values of these
    features for amino acids and then extract the corresponding feature group. You can extract them directly
    or use the tools that are introduced in the following papers. More information can also be found in the
    papers that are provided in the project-related module (they will also be introduced throughout the
    course and lectures). More information can be found in:
      - Ferdous_etal_2024
      - Muhammod_etal_2019
      - Chen_etal_2018
  
    2.2 Sequence-based features: These features are extracted based on the alphabetic sequence of
    peptides. To extract such features, we just rely on the ordering as well as the combination of alphabets
    (relying on statistical as well as mathematical models for feature extraction regardless of physicochemical
    property and nature of amino acids). Examples of such features are amino acid composition, occurrence,
    and bi-gram. More information can be found in:
    - Ferdous_etal_2024
    - Muhammod_etal_2019
    - Chen_etal_2018
   
    2.3 Protein encoding using Protein Language Models (PLMs): PLMs are Large Language Models (LLMs)
    that are trained on large protein databases and generate corresponding numerical encoding of proteins.
    Such encoding can be used as input features for different machine learning models.
  
    (More information on what PLM is and several different PLMs that you can use is provided in
    “Protein_Language_Model” that is provided in the “Project” module.
    You can use any of these PLMs (whichever you find more interesting, informative, or easier to use).

3. You will need to prepare the data (properly put them together with labels) and use different
machine learning methods on them (e.g., K-Nearest Neighbor (KNN), Support Vector Machine
(SVM), Naïve Bayes, Artificial Neural Network (ANN), Random Forest, Bagging). As the data
suggest, the problem is hand is a multi-class classification task.

    - You are required to use at least one deep learning architecture (e.g., Convolutional Neural Network
    (CNN), which is suitable when the input data is in matrix format). More information can be found in
    “Azim_etal_2023” in the group project module.
