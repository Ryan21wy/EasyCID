# EasyCID：Component Identiﬁcation with Raman Spectroscopy Made Easy

EasyCID is a practical and efficient tool for analyzing unknown mixtures with Raman spectroscopy. EasyCID provides a total workflow of component Identiﬁcation consisting of three functions: create the spectral database for managing Raman spectra, build the CNN model for each pure compound in spectral database, and identify the components in the unknown mixtures automatically and efficiently. In EasyCID, all the above functions can be easily implemented with the assistance of the graphical user interface (GUI).

## Installation

The current install version of EasyCID only supports Windows 64-bit version. It has been test on windows 7, windows 10.

Install Package: [EasyCID-1.0-Windows.exe]()

## Development version
  
1. Install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)   
2. Install [Git](https://git-scm.com/downloads)  
4. Open commond line, create environment and enter with the following commands:  

        conda create -n EasyCID python=3.7
        conda activate EasyCID

5. Clone the repository and enter:  

        git clone https://github.com/Ryan21wy/EasyCID.git
        cd EasyCID

6. Install dependency with the following commands:  
        
        pip install -r requirements.txt
        
7. Run MainCode.py:  

        python MainCode.py

## Usage

For the details on how to use EasyCID, please check [PDF ducomentation](). 

The html ducomentation and a demo are provided in the GUI of EasyCID.

## Contact

Wang Yue   
E-mail: ryanwy@csu.edu.cn 
