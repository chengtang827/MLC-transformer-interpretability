# MLC-extension

Will give you unaveraged weights for whatever type of attention and transformer module (encoder or decoder) you input: 
```python3 eval.py --fn_out_model net-BIML-top.pt --episode_type my_test --max --verbose --make_plots ```

Will give you averaged weights for whatever type of attention and transformer module (encoder or decoder) you input: 
```python3 eval.py --fn_out_model net-BIML-top.pt --episode_type my_test --max --verbose --make_plots --averaged_weights```

To retrieve models: 
```wget https://cims.nyu.edu/~brenden/supplemental/BIML-large-files/BIML_ml_models.zip```

To change query, edit text file in my_data/val/



