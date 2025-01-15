# EasyCID：Component Identiﬁcation with Raman Spectroscopy Made Easy

EasyCID is a practical and efficient tool for analyzing unknown mixtures with Raman spectroscopy. EasyCID provides a total workflow of component Identiﬁcation consisting of three functions: create the spectral database for managing Raman spectra, build the CNN model for each pure compound in spectral database, and identify the components in the unknown mixtures automatically and efficiently. In EasyCID, all the above functions can be easily implemented with the assistance of the graphical user interface (GUI).

## Installation

The current install version of EasyCID only supports Windows 64-bit version. It has been test on _**Windows 7**_, _**Windows 10**_ and _**Windows 11**_.

Install Package: [EasyCID-1.0.0-Windows.exe](https://github.com/Ryan21wy/EasyCID/releases/download/v1.0.0/EasyCID.exe)

#### Note: When installing, please do not change the default folder name (EasyCID).

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
1. Build Database & Import Data

https://user-images.githubusercontent.com/81405754/162189284-d9d3ba02-4e93-4a90-8b55-c9d8c7bdd58f.mov


2. Training CNN Models

https://user-images.githubusercontent.com/81405754/162186979-dc25a0f6-e675-417c-81b5-2f57c3b6d533.mov

3. Prediction

https://user-images.githubusercontent.com/81405754/162192498-dcc52b6c-6c55-4079-a909-b861b96b67a2.mov

4. Save Results

https://user-images.githubusercontent.com/81405754/162192883-2ba04376-5e7a-4d66-8ec7-f2f52194e464.mov

The Full video for using the EasyCID is available at the [video folder](https://github.com/Ryan21wy/EasyCID/blob/master/Videos).

For the details on how to use EasyCID, please check [PDF ducomentation](https://github.com/Ryan21wy/EasyCID/blob/master/Documentation.pdf). 

The html ducomentation and a demo are provided in the EasyCID GUI.

## Contact

Wang Yue   
E-mail: ryanwy@csu.edu.cn 
