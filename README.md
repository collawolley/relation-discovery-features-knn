# download and unzip data from Diego and Titov 2016 

```
wget https://www.dropbox.com/s/kqvqqh0eogqo1f4/Archive.zip?dl=0
mkdir data
mv Archive*  ./data/Archive.zip
unzip ./data/Archive.zip -d Archive

```

# Rerun Experiments: 

$ ipython

 
 >> from run_features import Run
 >> r = Run('./Archive/candidate-2000s.context.filtered.triples.pathfiltered.pos.single-relation.sortedondate.test.80%.txt') 
 >> r.run()








