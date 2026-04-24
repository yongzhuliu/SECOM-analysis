# first line: 1330
def _fit_resample_one(sampler, X, y, message_clsname="", message=None, params=None):
    with _print_elapsed_time(message_clsname, message):
        X_res, y_res = sampler.fit_resample(X, y, **params.get("fit_resample", {}))

        return X_res, y_res, sampler
