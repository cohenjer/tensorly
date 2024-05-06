1. Install ttm package for python by following instructions [here](https://github.com/bassoy/ttm/tree/main/ttmpy). You may have to edit the setup.py to add the correct path to openblas in the include field.
2. Also install the ttv package, available [here](https://github.com/bassoy/ttv).
3. Use the tensorly plugins to change the mode_dot implementation in order to support ttv and ttm
        'tl.plugins.use_bassoy_ttm()'
4. Enjoy hopefully faster mode_dot!
5. To revert the change: 'tl.plugins.use_default_ttm()'