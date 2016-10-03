# download and unzip data from Diego and Titov 2016 

```
wget https://www.dropbox.com/s/kqvqqh0eogqo1f4/Archive.zip?dl=0
mkdir data
mv Archive*  ./data/Archive.zip
unzip ./data/Archive.zip -d Archive

```

# Rerun Experiments: 

```
$ ipython
Python 2.7.6 (default, Jun 22 2015, 17:58:13) 
Type "copyright", "credits" or "license" for more information.

IPython 2.3.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
Warning: disable autoreload in ipython_config.py to improve performance.


 >> from run_features import Run
 >> r = Run('./Archive/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation.sortedondate.test.80%.txt') 
 >> r.run()

```






