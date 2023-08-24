
# Changes from Planet Specific

## Main Change: Grid Organisation

In each of the grids, a large number of models are generated from a set of varying parameters. In the Planet Specific Grid, there were 5 parameters, each witha few values they could take. Every combination of values had a model. Thus, the approach of the Planet Specific Grid fit was as follows:

- Iterate through all the files in the folder of models. Record the filenames.
- The filenames have a specific nomenclature, so the model parameters are written in order separated by underscores. This means the program can extract all the different parameters from every file, leaving it with 5 lists of values used in the model.
- Given that the assumption that every combination of parameters is used is true, the code can safely iterate through these lists and reconstruct the filename.
- It then uses `scipy.optimize()` to minimise systematic error and determine $\chi^2$ for each model.
- From this it determines the probability density function.

However, in the Self Consistent Grid not every combination of model parameters is used. Also, there are now only 3 parameters. This is illustrated here in the beginning of [HD-209458's grid](/testgrid_sc/HD-209458/):

| recirculation factor | log(metalicity) | carbon/oxygen ratio |
|-------|------|------|
|0.25|-1.0|0.35|
|0.25|-1.0|1.00|
|0.25|-1.0|1.50|
|0.25|0.0|0.35|
|0.25|0.0|0.55|
|0.25|0.0|0.70|
|0.25|0.0|0.75|
|0.25|0.0|1.00|
|0.25|1.0|0.35|
|...|...|...|

Therefore, the method that [Self Consistent Grid fit](/SCgrid_fit_JY.py) must differ from the Planet Specific Grid fit program. In PS Grid fit, arrays are initilised before the loop runs for storing the data and $\chi^2$ values. In these arrays there is an axis for each parameter. If this same approach is used for the SC Grid it would end up with many black spaces and would be confusing and cumbersome.

Instead, [Self Consistent Grid fit](/SCgrid_fit_JY.py) determines $\chi^2$ as it iterates through the files for the first time and then later deals with the parameters from the model filenames. This is an overview of the approach:

- Iterate through all the files in the folder of models. Extract the parameters as before. Find $\chi^2$ within this loop. All the relevant details are appended to arrays.

- Now we are left with 1-D arrays of the model parameters and fit info. They are in an arbitrary order but critically are all in the same order.

- The same mathematics is carried out on the arrays to find the best fitting model. This has also been expedited by avoiding needing to iterate through every parameter; instead bulky loops have been replaced with single line generators. Compare:
    ```python
    #ps
    for temp in range(0, ntemp):
        for metal in range(0, nmetal):
            for co in range(0, nco):
                for haze in range(0, nhaze):
                    for cloud in range(0, ncloud):
                        norm_prob_density[temp,metal,co,haze,cloud] = np.exp(-0.5 * model_chi[temp,metal,co,haze,cloud] - log_norm_grid
                        
    #sc
    norm_prob_density = [np.exp(-0.5 * chi - log_norm_grid) for chi in model_chi]
    ```

## Other Changes

One area where the PS Grid fit was not user friendly was the calculation the dx values. These are essentially the 'widths' of each discrete value a parameter can take. They are calculated by taking the midpoints between the values (at the top and bottom these are assumed symmetric) and then finding their difference. One complexity is that certain variables may be limited to $0\le x\le1$ so this needs to be accounted for in the midpoint calculation for the top and bottom.

In PS Grid fit this was manually entered as 5 arrays, each containing the same number of elements as there were values for each parameter.

Now this is automated by the [`get_bin_widths()`](https://github.com/BabelFish0/matching-atmospheres/blob/42eb49466befba93493221d481f88201b8165c71/SCgrid_fit_JY.py#L66-L85) function. This uses NumPy arrays to neatly find the midpoints and also clip the top and bottom if needed. This has [its own testing file](/tools/dx_finder.py) for trying out numbers.