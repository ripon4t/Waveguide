"""Small helpers shared by the Lumerical (lumapi) scripts."""
import glob
import os
import sys


def import_lumapi():
    """Import lumapi, searching the usual install locations if it is not on the path.

    Set the environment variable LUMAPI_PATH to the folder that contains lumapi.py
    (e.g. C:\\Program Files\\Lumerical\\v241\\api\\python) to override the search.
    """
    try:
        import lumapi  # noqa: F401
        return lumapi
    except ImportError:
        pass
    candidates = []
    if os.environ.get("LUMAPI_PATH"):
        candidates.append(os.environ["LUMAPI_PATH"])
    candidates += sorted(glob.glob(r"C:\Program Files\Lumerical\v*\api\python"), reverse=True)
    candidates += sorted(glob.glob(r"C:\Program Files\ANSYS Inc\v*\Lumerical\api\python"), reverse=True)
    candidates += sorted(glob.glob("/opt/lumerical/v*/api/python"), reverse=True)
    candidates += sorted(glob.glob("/Applications/Lumerical*/Contents/API/Python"), reverse=True)
    for path in candidates:
        if os.path.isfile(os.path.join(path, "lumapi.py")):
            sys.path.append(path)
            import lumapi
            return lumapi
    raise ImportError(
        "Could not find lumapi. Install Ansys Lumerical and either add its api/python "
        "folder to PYTHONPATH or set LUMAPI_PATH to that folder."
    )


def set_props(sim, props, label=""):
    """Set many properties on the currently selected object.

    A property name that does not exist in your Lumerical version is reported and
    skipped instead of aborting the whole script, so you can see exactly which
    names need adjusting (check the object's Properties tab / Object Library).
    """
    for key, value in props.items():
        try:
            sim.set(key, value)
        except Exception as exc:  # lumapi raises LumApiError
            print(f"[warning] {label}: could not set '{key}' -> {value!r}: {exc}")


def try_values(sim, key, values, label=""):
    """Try several candidate values for one property until one is accepted."""
    for v in values:
        try:
            sim.set(key, v)
            return v
        except Exception:
            continue
    print(f"[warning] {label}: none of {values!r} accepted for '{key}'")
    return None
