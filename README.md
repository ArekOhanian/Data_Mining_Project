In order to run project, download ZIP file of all code, extract the contents. Open in an IDE, and make sure you have all the imports installed, i.e. pip install groq, pip install eel, etc.. Then you can run the 'main.py' file and that will pull up the browser. Browser can be used and results will also pop up in the terminal.

=>You can run this command in mac terminal to set up the virtual environment and run the main file to use the browser. It makes sure everything is installed correctly
cd "/Users/your/path/here
python3 -m venv venv --clear && \
source venv/bin/activate && \
pip install --upgrade pip && \
pip install groq eel pandas matplotlib seaborn scikit-learn numpy mlxtend && \
python main.py