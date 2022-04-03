# Add repo to install the last R
sudo apt install apt-transport-https software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu eoan-cran35/'

# Install R
sudo apt update
sudo apt install gdebi libxml2-dev libssl-dev libcurl4-openssl-dev libopenblas-dev r-base r-base-dev

# Install RStudio
cd ~/Downloads
wget https://download1.rstudio.org/desktop/bionic/amd64/rstudio-1.2.5001-amd64.deb
sudo gdebi rstudio-1.2.5001-amd64.deb
#~ # I really don't know if this is necessary anymore. 
#~ test -r ~/.bash_profile && printf '\nexport QT_STYLE_OVERRIDE=gtk\n' | sudo tee -a ~/.bash_profile
#~ test -r ~/.zshrc && printf '\nexport QT_STYLE_OVERRIDE=gtk\n' | sudo tee -a ~/.zshrc
#~ printf '\nexport QT_STYLE_OVERRIDE=gtk\n' | sudo tee -a ~/.profile

#~ # Install common packages
#~ R --vanilla << EOF
#~ install.packages(c("tidyverse","data.table","dtplyr","devtools","roxygen2","bit64","readr"), repos = "https://cran.rstudio.com/")
#~ q()
#~ EOF

#~ # Install TDD packages
#~ R --vanilla << EOF
#~ install.packages("testthis")
#~ q()
#~ EOF

#~ # Export to HTML/Excel
#~ R --vanilla << EOF
#~ install.packages(c("htmlTable","openxlsx"), repos = "https://cran.rstudio.com/")
#~ q()
#~ EOF

#~ # Blog tools
#~ R --vanilla << EOF
#~ install.packages(c("knitr","rmarkdown"), repos='http://cran.us.r-project.org')
#~ q()
#~ EOF
#~ sudo apt install python-pip
#~ sudo apt install python3-pip
#~ sudo -H pip install markdown rpy2==2.7.1 pelican==3.7.1
#~ sudo -H pip3 install markdown rpy2==2.9.3 pelican==3.7.1 

#~ # PDF extraction tools
#~ sudo apt install libpoppler-cpp-dev default-jre default-jdk r-cran-rjava
#~ sudo R CMD javareconf
#~ R --vanilla << EOF
#~ library(devtools)
#~ install.packages("pdftools", repos = "https://cran.rstudio.com/")
#~ install_github("ropensci/tabulizer")
#~ q()
#~ EOF

#~ # TTF/OTF fonts usage
#~ sudo apt install libfreetype6-dev
#~ R --vanilla << EOF
#~ install.packages("showtext", repos = "https://cran.rstudio.com/")
#~ q()
#~ EOF

#~ # Cairo for graphic devices
#~ sudo apt install libgtk2.0-dev libxt-dev libcairo2-dev
#~ R --vanilla << EOF
#~ install.packages("Cairo", repos = "https://cran.rstudio.com/")
#~ q()
#~ EOF

#~ # Texlive for Latex/knitr
#~ sudo apt -y install texlive
#~ sudo apt -y install texlive-latex-recommended texlive-pictures texlive-latex-extra
