## Author: Md Mahmudur Rahman
## Version : 1.0
## Date: Feb. 4, 2021


from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf

# Change the directory 'dir' where the code file is stored
#Example for dir: "./DeepPseudo_AAAI/Marginal_DeepPseudo"
os.chdir('dir')

os.system('python Data_Preprocessing.py')

os.system('python RandomSearch.py')

os.system('python summarize_results.py')
