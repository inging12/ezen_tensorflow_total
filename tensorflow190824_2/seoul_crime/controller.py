from seoul_crime.cctv_pop import CCTVModel
from seoul_crime.crime_police import CrimeModel
from seoul_crime.Folium_test import FoliumTest
from seoul_crime.police_norm import PoliceNormalModel

class Controller:
    def __init__(self):
        self._cctv = CCTVModel()
        self._crime = CrimeModel()
        self._usa = FoliumTest()
        self._police_norm = PoliceNormalModel()

    def execute(self):
        #cctv = self._cctv
        #cctv.hook_process()

        #crime = self._crime
        #crime.hook_process()

        #usa = self._usa
        #usa.hook()

        pn = self._police_norm
        pn.hook()