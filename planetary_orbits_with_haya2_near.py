#python
# -*- coding:utf-8 -*-

import math
import re
from datetime import datetime, timedelta, timezone
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines



"""
DATA 1800AD - 2050AD equinox of J2000(JD2451545.0) from http://ssd.jpl.nasa.gov/txt/p_elem_t1.txt
               a              e               I                L            long.peri.      long.node.
           AU, AU/Cy     rad, rad/Cy     deg, deg/Cy      deg, deg/Cy      deg, deg/Cy     deg, deg/Cy
-----------------------------------------------------------------------------------------------------------
Mercury   0.38709927      0.20563593      7.00497902      252.25032350     77.45779628     48.33076593
          0.00000037      0.00001906     -0.00594749   149472.67411175      0.16047689     -0.12534081
Venus     0.72333566      0.00677672      3.39467605      181.97909950    131.60246718     76.67984255
          0.00000390     -0.00004107     -0.00078890    58517.81538729      0.00268329     -0.27769418
EM Bary   1.00000261      0.01671123     -0.00001531      100.46457166    102.93768193      0.0
          0.00000562     -0.00004392     -0.01294668    35999.37244981      0.32327364      0.0
Mars      1.52371034      0.09339410      1.84969142       -4.55343205    -23.94362959     49.55953891
          0.00001847      0.00007882     -0.00813131    19140.30268499      0.44441088     -0.29257343
Jupiter   5.20288700      0.04838624      1.30439695       34.39644051     14.72847983    100.47390909
         -0.00011607     -0.00013253     -0.00183714     3034.74612775      0.21252668      0.20469106
Saturn    9.53667594      0.05386179      2.48599187       49.95424423     92.59887831    113.66242448
         -0.00125060     -0.00050991      0.00193609     1222.49362201     -0.41897216     -0.28867794
Uranus   19.18916464      0.04725744      0.77263783      313.23810451    170.95427630     74.01692503
         -0.00196176     -0.00004397     -0.00242939      428.48202785      0.40805281      0.04240589
Neptune  30.06992276      0.00859048      1.77004347      -55.12002969     44.96476227    131.78422574
          0.00026291      0.00005105      0.00035372      218.45945325     -0.32241464     -0.00508664
Pluto    39.48211675      0.24882730     17.14001206      238.92903833    224.06891629    110.30393684
         -0.00031596      0.00005170      0.00004818      145.20780515     -0.04062942     -0.01183482


DATA 3000BC - 3000AD equinox of J2000(JD2451545.0) from http://ssd.jpl.nasa.gov/txt/p_elem_t2.txt
               a              e               I                L            long.peri.      long.node.
           AU, AU/Cy     rad, rad/Cy     deg, deg/Cy      deg, deg/Cy      deg, deg/Cy     deg, deg/Cy
------------------------------------------------------------------------------------------------------
Mercury   0.38709843      0.20563661      7.00559432      252.25166724     77.45771895     48.33961819
          0.00000000      0.00002123     -0.00590158   149472.67486623      0.15940013     -0.12214182
Venus     0.72332102      0.00676399      3.39777545      181.97970850    131.76755713     76.67261496
         -0.00000026     -0.00005107      0.00043494    58517.81560260      0.05679648     -0.27274174
EM Bary   1.00000018      0.01673163     -0.00054346      100.46691572    102.93005885     -5.11260389
         -0.00000003     -0.00003661     -0.01337178    35999.37306329      0.31795260     -0.24123856
Mars      1.52371243      0.09336511      1.85181869       -4.56813164    -23.91744784     49.71320984
          0.00000097      0.00009149     -0.00724757    19140.29934243      0.45223625     -0.26852431
Jupiter   5.20248019      0.04853590      1.29861416       34.33479152     14.27495244    100.29282654
         -0.00002864      0.00018026     -0.00322699     3034.90371757      0.18199196      0.13024619
Saturn    9.54149883      0.05550825      2.49424102       50.07571329     92.86136063    113.63998702
         -0.00003065     -0.00032044      0.00451969     1222.11494724      0.54179478     -0.25015002
Uranus   19.18797948      0.04685740      0.77298127      314.20276625    172.43404441     73.96250215
         -0.00020455     -0.00001550     -0.00180155      428.49512595      0.09266985      0.05739699
Neptune  30.06952752      0.00895439      1.77005520      304.22289287     46.68158724    131.78635853
          0.00006447      0.00000818      0.00022400      218.46515314      0.01009938     -0.00606302
Pluto    39.48686035      0.24885238     17.14104260      238.96535011    224.09702598    110.30167986
          0.00449751      0.00006016      0.00000501      145.18042903     -0.00968827     -0.00809981
------------------------------------------------------------------------------------------------------
"""

class Planet:
    def __init__(self, name):
        self.name = name
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []
        self.angleEVE = 0

        if self.name == "Mercury":
            self.a = 0.38709927
            self.e = 0.20563593
            self.varpi0 = 77.45779628
            self.varpi_dot = 0.16047689
            self.L0 = 252.25032350
            self.L_dot = 149472.67486623
            self.I0 = 7.00497902
            self.I_dot = -0.00594749
            self.Omega0 = 48.33076593
            self.Omega_dot = -0.12534081
        elif self.name == "Venus":
            self.a = 0.72333566
            self.e = 0.00677672
            self.varpi0 = 131.60246718
            self.varpi_dot = 0.00268329
            self.L0 = 181.97909950
            self.L_dot = 58517.81538729
            self.I0 = 3.39467605
            self.I_dot = -0.00078890
            self.Omega0 = 76.67984255
            self.Omega_dot = -0.27769418
        elif self.name == "Earth": # EM Bary
            self.a = 1.00000261
            self.e = 0.01671123
            self.varpi0 = 102.93768193
            self.varpi_dot = 0.32327364
            self.L0 = 100.46457166
            self.L_dot = 35999.37244981
            self.I0 = -0.00001531
            self.I_dot = -0.01294668
            self.Omega0 = 0.0
            self.Omega_dot = 0.0
        elif self.name == "Mars":
            self.a = 1.52371034
            self.e = 0.09339410
            self.varpi0 = -23.94362959
            self.varpi_dot = 0.44441088
            self.L0 = - 4.55343205
            self.L_dot = 19140.30268499
            self.I0 = 1.84969142
            self.I_dot = -0.00813131
            self.Omega0 = 49.55953891
            self.Omega_dot = -0.29257343
        elif self.name == "Jupiter":
            self.a = 5.20288700
            self.e = 0.04838624
            self.varpi0 = 14.72847983
            self.varpi_dot = 0.21252668
            self.L0 = 34.39644051
            self.L_dot = 3034.74612775
            self.I0 = 1.30439695
            self.I_dot = -0.00183714
            self.Omega0 = 100.47390909
            self.Omega_dot = 0.20469106
        elif self.name == "Saturn":
            self.a = 9.53667594
            self.e = 0.05386179
            self.varpi0 = 92.59887831
            self.varpi_dot = -0.41897216
            self.L0 = 49.95424423
            self.L_dot = 1222.49362201
            self.I0 = 2.48599187
            self.I_dot = 0.00193609
            self.Omega0 = 113.66242448
            self.Omega_dot = -0.28867794
        elif self.name == "Uranus":
            self.a = 19.18916464
            self.e = 0.04725744
            self.varpi0 = 170.95427630
            self.varpi_dot = 0.40805281
            self.L0 = 313.23810451
            self.L_dot = 428.48202785
            self.I0 = 0.77263783
            self.I_dot = -0.00242939
            self.Omega0 = 74.01692503
            self.Omega_dot = 0.04240589
        elif self.name == "Neptune":
            self.a = 30.06992276
            self.e = 0.00859048
            self.varpi0 = 44.96476227
            self.varpi_dot = -0.32241464
            self.L0 = -55.12002969
            self.L_dot = 218.45945325
            self.I0 = 1.77004347
            self.I_dot = 0.00035372
            self.Omega0 = 131.78422574
            self.Omega_dot = -0.00508664
        elif self.name == "Pluto":      # from JPL
            # perihelion distance[q] : 29.658
            # aperihelion distance[Q]: 49.306
            self.a = 39.48211675
            self.e = 0.24882730
            self.varpi0 = 224.06891629
            self.varpi_dot = -0.04062942
            self.L0 = 238.92903833
            self.L_dot = 145.20780515
            self.I0 = 17.14001206
            self.I_dot = 0.00004818
            self.Omega0 = 110.30393684
            self.Omega_dot = -0.01183482

            # # value in long term
            # self.a = 39.48686035
            # self.e = 0.24885238
            # self.varpi0 = 224.09702598
            # self.varpi_dot = -0.00968827
            # self.L0 = 238.96535011
            # self.L_dot = 145.18042903
            # self.I0 = 17.14104260
            # self.I_dot = 0.00000501
            # self.Omega0 = 110.30167986
            # self.Omega_dot = -0.00809981

        AU = 149597870691
        GM_SUN = 1.32712440018 * 10 ** 20
        K0 = math.sqrt(GM_SUN / (AU * AU * AU)) * 86400
        self.n = K0 * self.a ** (-1.5) # [rad/day]
        self.b = self.a * math.sqrt(1 - self.e * self.e)

        # convert deg -> rad
        self.varpi0 = math.radians(self.varpi0)
        self.varpi_dot = math.radians(self.varpi_dot)
        self.L0 = math.radians(self.L0)
        self.L_dot = math.radians(self.L_dot)
        self.I0 = math.radians(self.I0)
        self.I_dot = math.radians(self.I_dot)
        self.Omega0 = math.radians(self.Omega0)
        self.Omega_dot = math.radians(self.Omega_dot)


    def calc(self, jd, params):
        '''
        calculate target planet position
        '''

        T = jd / 36525.0        # Julian Century

        self.varpi = self.varpi0 + self.varpi_dot * T
        self.L     = self.L0     + self.L_dot * T
        self.I     = self.I0     + self.I_dot * T
        self.Omega = self.Omega0 + self.Omega_dot * T
        self.omega = self.varpi - self.Omega
        self.M = self.L - self.varpi
        self.E = self.solveE(self.M, self.e)

        x_prime = self.a * (math.cos(self.E) - self.e)
        y_prime = self.b * math.sin(self.E)
        r_prime = math.sqrt(x_prime ** 2 + y_prime ** 2)

        mat = self.getMatrix(self.I, self.Omega, self.omega)

        self.x = mat[0][0] * x_prime + mat[0][1] * y_prime
        self.y = mat[1][0] * x_prime + mat[1][1] * y_prime
        self.z = mat[2][0] * x_prime + mat[2][1] * y_prime

        self.r = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        self.r_xy = math.sqrt(self.x*self.x + self.y*self.y)

        # convert x, y, z pos: px, py, pz by theta, phi
        self.px, self.py, self.pz = self.convertCood(self.x, self.y, self.z, params)

        self.xl.append(self.px)
        self.yl.append(self.py)
        self.zl.append(self.pz)
        self.rl.append(self.r)

        self.r_xyl.append(self.r_xy)


    def getMatrix(self, I, Omega, omega):
        # reference at HoshizoraYokochou[http://hoshizora.yokochou.com/calculation/orbit.html]
        # convert cord.[orbit surface] -> cord.[center of Sun surface]

        cos_I = math.cos(I)
        sin_I = math.sin(I)
        cos_Omega = math.cos(Omega)
        sin_Omega = math.sin(Omega)
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)

        M11 =  cos_omega * cos_Omega - sin_omega * sin_Omega * cos_I
        M12 = -sin_omega * cos_Omega - cos_omega * sin_Omega * cos_I
        M21 =  cos_omega * sin_Omega + sin_omega * cos_Omega * cos_I
        M22 = -sin_omega * sin_Omega + cos_omega * cos_Omega * cos_I
        M31 =  sin_omega * sin_I
        M32 =  cos_omega * sin_I

        return [[M11, M12],
                [M21, M22],
                [M31, M32]]


    def solveE(self, M, e):
        count = 1
        E1 = M
        while True:
            # E2 = M + math.degrees(e * math.sin(math.radians(E1)))
            delta_E = (M - E1 + e * math.sin(E1)) / (1 - e * math.cos(E1))
            E2 = E1 + delta_E
            E1 = E2

            if math.fabs(delta_E) < 0.00000001:
                break
            if count % 1000 == 0:
                print("count of solveE: {}".format(count))
            if count > 10000: exit(0)
            count += 1
        return E2


    def convertCood(self, x, y, z, params):
        theta, phi = params['theta'], params['phi']
        # rotate theta around x-axis
        _theta = math.radians(theta)
        x2 = x
        y2 = y * math.cos(_theta) - z * math.sin(_theta)
        z2 = y * math.sin(_theta) + z * math.cos(_theta)

        # rotate phi around y-axis
        _phi = math.radians(phi)
        x3 = x2 * math.cos(_phi) + z2 * math.sin(_phi)
        y3 = y2
        z3 = z2 * math.cos(_phi) - x2 * math.sin(_phi)

        return [x3, y3, z3]


    def toJD(self, dt):
        '''
        from http://astronomy.webcrow.jp/time/gregoriancalendar-julianday.html
        '''
        y,m,d,H,M,S = dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second

        if m < 3:
            m = m + 12
            y = y - 1

        jd_year = int(365.25 * y) - int(y/100.) + int(y/400.) + 1721088.5
        jd_day = int(30.59 * (m - 2)) + int(d) + H/24. + M/1440. + S/86400.
        jd = jd_year + jd_day
        return jd


    def drawOrbit(self, ax, params):
        '''
        plot line of target planet (for planet orbit)
        '''
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []

        begin_time = datetime(2015,1,1, tzinfo=timezone.utc)
        end_time = datetime(2016,1,1, tzinfo=timezone.utc)
        interval_time = timedelta(days=self.a)

        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = begin_time if end_time==None else end_time.replace(tzinfo=timezone.utc)
        days_interval = timedelta(days=1) if interval_time==None else interval_time
        target_time = begin_time

        while target_time <= end_time:
            jd = self.toJD(target_time) - 2451545.0 # J2000
            self.calc(jd, params)
            target_time += interval_time

        ax.plot(self.xl, self.yl, '-', lw=0.5,
                color = {'Mercury': 'b',
                         'Venus'  : '#ffd700',
                         'Earth'  : 'g',
                         'Mars'   : 'r',
                         'Jupiter': '#8b4512',
                         'Saturn' : '#deb887',
                         'Uranus' : '#40e0d0',
                         'Neptune': '#00bfff',
                         'Pluto'  : 'k'
                        }[self.name])


    def plotPoint(self, ax, params, begin_time, end_time = None, interval_time = None):
        '''
        plot point of target planet on target date
        '''
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []

        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = begin_time if end_time==None else end_time.replace(tzinfo=timezone.utc)
        interval_time = timedelta(days=1) if interval_time==None else interval_time
        target_time = begin_time

        while target_time <= end_time:
            jd = self.toJD(target_time) - 2451545.0 # J2000
            self.calc(jd, params)
            target_time += interval_time

        ax.plot(self.xl, self.yl, '*', ms=8,
                color = {'Mercury': 'b',
                         'Venus'  : '#ffd700',
                         'Earth'  : 'g',
                         'Mars'   : 'r',
                         'Jupiter': '#8b4513',
                         'Saturn' : '#deb887',
                         'Uranus' : '#40e0d0',
                         'Neptune': '#00bfff',
                         'Pluto'  : 'gray'
                     }[self.name])

    def textDate(self, ax, params, begin_time, end_time=None, interval_time=None):
        '''
        caption text of planets
        '''
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []
        dl = []

        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = begin_time if end_time==None else end_time.replace(tzinfo=timezone.utc)
        interval_time = timedelta(days=1) if interval_time==None else interval_time
        target_time = begin_time

        while target_time <= end_time:
            jd = self.toJD(target_time) - 2451545.0 # J2000
            self.calc(jd, params)
            dl.append(target_time)
            target_time += interval_time

        for i in range(len(dl)):
            ax.text(self.xl[i], self.yl[i],
                    "${0:%m/%d}^{{\mathrm{{'}}{0:%y}}}$".format(dl[i]), fontsize=6, ha='left', va='top')


    def calcEVE(self, ax, params):
        '''
        for exhibition function
        '''
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []

        jd = self.toJD(params['EVEday']) - 2451545.0 # J2000
        self.calc(jd, params)

        return [self.xl, self.yl]


    def plotPointOnEVE(self, ax, params):
        '''
        for exhibition function
        '''
        ax.plot(params['EVE'][0], params['EVE'][1], 'D',
                 color = {'Earth'  : '#00fa9a'}[self.name])


    def textAngleEVE(self, ax, params, begin_time, end_time=None, interval_time=None):
        '''
        for exhibition function
        angle of Vernal Equinox day's Earth positon <-> Sun position <-> Planet position
        '''
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []

        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = begin_time if end_time==None else end_time.replace(tzinfo=timezone.utc)
        interval_time = timedelta(days=1) if interval_time==None else interval_time
        target_time = begin_time

        while target_time <= end_time:
            jd = self.toJD(target_time) - 2451545.0 # J2000
            self.calc(jd, params)
            target_time += interval_time

        for i in range(len(self.xl)):
            a = np.array([self.xl[i], self.yl[i]])
            b = np.array(params['EVE'])

            ang = self.angle2vector(a, b) # ang[radian]
            ang_deg = math.degrees(ang)

            print("{:s} angle(x-y): {:.3f}".format(self.name, ang_deg))
            ax.text(self.xl[i], self.yl[i], "{:.1f}".format(ang_deg), fontsize=6, ha='left', va='bottom')


    '''
    misc functions
    '''
    def printDistanceQq(self):
        print("-------------------")
        print(self.name)
        print(" perihelion distance[q] : {:.3f}\n aperihelion distance[Q]: {:.3f}".format(min(self.rl), max(self.rl)))
        print("-------------------")

    def printDistance(self):
        print("*********")
        print(self.name)
        print(" distance from Sun: {:.3f}".format(self.r))
        print("*********")

    def printDistanceXY(self):
        print("{:s} distance from Sun(x-y): {:.3f}".format(self.name, self.distance2point(np.array([self.px, self.py]), np.array([0, 0]))))

    def distance2point(self, a, b):       # a,b: numpy array([x1,y1], [x2,y2])
        global mag
        u = b - a
        return np.linalg.norm(u * mag)

    def angle2vector(self, a, b):           # a,b: vector (numpy)
        cos_ang = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
        ang = math.acos(cos_ang)
        return ang                  # radian


class JAXA(Planet):
    def __init__(self, name, data_file):
        self.name = name

        if self.name == "Hayabusa2":
            self.data = OrderedDict()
            h2list = [x.strip() for x in open(data_file,'r',encoding='utf-8').readlines()]
            for line in h2list:
                if re.match('^#', line): continue
                _d = {}
                c = re.split('\s+', line)

                _date = datetime.strptime(c[0], "%Y/%m/%d.%H:%M:%S")
                _d['date'] = datetime(_date.year, _date.month, _date.day, _date.hour, _date.minute, _date.second,  tzinfo=timezone.utc) # date type
                _d['lp']   = float(c[1])     # L+ [days]
                _d['x']  = float(c[2])       # X pos. [au]
                _d['y']  = float(c[3])       # Y pos. [au]
                _d['z']  = float(c[4])       # Z pos. [au]
                _d['ex'] = float(c[5])
                _d['ey'] = float(c[6])
                _d['ez'] = float(c[7])
                _d['rs'] = float(c[11])      # distance of Sun-Haya2 [10**4 km]
                _d['re'] = float(c[12])      # distance of Earth-Haya2 [10**4 km]
                _d['ra'] = float(c[13])      # distance of 1999JU3-Haya2 [10**4 km]
                _d['vs']  = float(c[14])     # velocity of Haya2 on Sun [km/sec]
                _d['ve']  = float(c[15])     # velocity of Haya2 on Earth [km/sec]
                _d['alpha'] = float(c[16])   # ra [deg]
                _d['delta'] = float(c[17])   # dec [deg]
                _d['Dflt']  = float(c[18])   # distance of fling [10**4 km]

                self.data["{:%Y%m%d%H%M%S}".format(_d['date'])] = _d

        if self.name == "Ryugu":
            self.data = OrderedDict()
            h2list = [x.strip() for x in open(data_file,'r',encoding='utf-8').readlines()]
            for line in h2list:
                if re.match('^#', line): continue
                _d = {}
                c = re.split('\s+', line)

                _date = datetime.strptime(c[0], "%Y/%m/%d.%H:%M:%S")
                _d['date'] = datetime(_date.year, _date.month, _date.day, _date.hour, _date.minute, _date.second,  tzinfo=timezone.utc) # date type
                _d['lp']   = float(c[1])      # L+ [days]
                _d['x']  = float(c[8])        # X pos. [au]
                _d['y']  = float(c[9])        # Y pos. [au]
                _d['z']  = float(c[10])       # Z pos. [au]
                _d['ra'] = float(c[13])       # distance of Ryugu-Haya2 [10**4 km]

                self.data["{:%Y%m%d%H%M%S}".format(_d['date'])] = _d


    def drawOrbitJAXA(self, ax, params, begin_time=None, end_time=None, interval_time=None):
        '''
        plot line of target planet (for planet orbit)
        '''
        xl, yl = [], []
        for k,v in self.data.items():
            px, py, pz = self.convertCood(v['x'], v['y'], v['z'], params)
            xl.append(px)
            yl.append(py)

        ax.plot(xl, yl, '--', lw=0.5,
                color = {'Hayabusa2' : '#00bfff',
                         'Ryugu': 'gray'
                     }[self.name])

    def plotPointJAXA(self, ax, params, begin_time=None, end_time=None, interval_time=None):
        '''
        plot point of target planet on target date
        '''
        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = end_time.replace(tzinfo=timezone.utc)
        target_time = begin_time

        while target_time <= end_time:
            k = "{:%Y%m%d%H%M%S}".format(target_time)
            if k in self.data:
                v = self.data[k]
                px, py, pz = self.convertCood(v['x'], v['y'], v['z'], params)
                ax.plot(px, py,
                            {'Hayabusa2' : 'v',
                             'Ryugu': 'p'
                         }[self.name],
                            ms=6,
                            color = {'Hayabusa2' : '#00bfff',
                                     'Ryugu': 'gray'
                                 }[self.name])
            target_time += interval_time


    def textDateJAXA(self, ax, params, begin_time=None, end_time=None, interval_time=None):
        '''
        caption text of planets
        '''
        begin_time = begin_time.replace(tzinfo=timezone.utc)
        end_time = end_time.replace(tzinfo=timezone.utc)
        target_time = begin_time

        while target_time <= end_time:
            k = "{:%Y%m%d%H%M%S}".format(target_time)
            if k in self.data:
                v = self.data[k]
                px, py, pz = self.convertCood(v['x'], v['y'], v['z'], params)
                ax.text(px, py,
                        "${0:%m/%d}^{{\mathrm{{'}}{0:%y}}}$".format(v['date']),
                        # "${:%H:%M:%S}$".format(v['date']),
                        fontsize=6, ha='left', va='top')
            target_time += interval_time


    def textAngleEVEJAXA(self, ax, params, begin_time, end_time=None, interval_time=None):
        '''
        for exhibition function
        angle of Vernal Equinox day's Earth positon <-> Sun position <-> Planet position
        '''
        end_time = begin_time if end_time==None else end_time
        interval_time = timedelta(days=1) if interval_time==None else interval_time

        while target_time <= end_time:
            k = "{:%Y%m%d%H%M%S}".format(target_time)
            if k in self.data:
                v = self.data[k]
                px, py, pz = self.convertCood(v['x'], v['y'], v['z'], params)
                a = np.array([px, py])
                b = np.array(params['EVE'])
                ang = self.angle2vector(a, b) # ang[radian]
                ang_deg = math.degrees(ang)
                print("{:s} angle(x-y): {:.3f}".format(self.name, ang_deg))
                ax.text(px, py, "{:.1f}".format(ang_deg), fontsize=6, ha='left', va='bottom')
            target += interval_time


def main():
    params = {'inner': None,
              # 'theta': 6,
              # 'phi': -8,
              'theta': 0,
              'phi': 0,
              'mag': 0.245/1.,
              'EVE': [0, 0],
              'EVEday': datetime(2014, 3, 21, tzinfo=timezone.utc)}

    target_date = datetime(2015, 12, 3, tzinfo=timezone.utc) # plot position on target_date

    ####### INNER ########
    params['inner'] = True

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(111)
    ax.set_xlabel("$x[\mathrm{au}]$")
    ax.set_ylabel("$y[\mathrm{au}]$")
    ax.axis('equal')
    if params['inner']:
        # ax.axis([-2,2,-2,2])
        # ax.axis([0.3, 0.4, 0.8, 1.0])
        ax.axis([0.325, 0.3270, 0.9297, 0.9317])
        # ax.axis([0.306, 0.310, 0.92, 0.93])
        # ax.set_title('inner')
    else:
        ax.axis([-5,5,-5,5])
        ax.set_title('outer')

    ax.plot(0, 0, "ro") # SUN

    Earth = Planet("Earth")
    params['EVE'] = Earth.calcEVE(ax, params)
    Earth.plotPointOnEVE(ax, params) # plot Equinox of Earth J2000

    Mercury = Planet("Mercury")
    Venus = Planet("Venus")
    Mars = Planet("Mars")
    Jupiter = Planet("Jupiter")
    Saturn = Planet("Saturn")
    Uranus = Planet("Uranus")
    Neptune = Planet("Neptune")
    Pluto = Planet("Pluto")

    Earth.drawOrbit(ax, params)
    Earth.plotPoint(ax, params, target_date)
    Earth.textDate(ax, params, target_date)
    Earth.textAngleEVE(ax, params, target_date)

    target_date = datetime(2015,12,3,9,30, tzinfo=timezone.utc)
    Earth.plotPoint(ax, params, target_date)

    target_date = datetime(2015,12,3,10,0, tzinfo=timezone.utc)
    Earth.plotPoint(ax, params, target_date)

    target_date = datetime(2015,12,3,10,30, tzinfo=timezone.utc)
    Earth.plotPoint(ax, params, target_date)

    target_date = datetime(2015,12,4, tzinfo=timezone.utc)
    Earth.plotPoint(ax, params, target_date)
    Earth.textDate(ax, params, target_date)
    Earth.textAngleEVE(ax, params, target_date)

    ### JAXA ###
    ## constructor : JAXA(nameStr, data_file)
    Haya2 = JAXA("Hayabusa2", "hy2_trj_EC_10min.txt")
    Ryugu = JAXA("Ryugu", "hy2_trj_EC_10min.txt")

    ## JAXA.drawOrbitJAXA(axisObj, paramsDic, begin_lp, end_lp[opt])
    Haya2.drawOrbitJAXA(ax, params)
    # Ryugu.drawOrbitJAXA(ax, params)

    ## JAXA.plotPointJAXA(axisObj, paramsDic, begin_lp, end_lp[opt], days_interval[opt])
    Haya2.plotPointJAXA(ax, params, datetime(2015, 12, 3, 9, 30, 0), datetime(2015, 12, 3, 10, 30, 0), timedelta(minutes=10))
    # Ryugu.plotPointJAXA(ax, params)

    # ## JAXA.textDateJAXA(axisObj, paramsDic, begin_lp, end_lp[opt], days_interval[opt])
    Haya2.textDateJAXA(ax, params, datetime(2015, 12, 3, 9,30,0), datetime(2015, 12, 3, 10,30,0), timedelta(minutes=30))
    # Ryugu.textDateJAXA(ax, params, 365)

    # ## JAXA.textAngleEVEJAXA(axisObj, paramsDic, begin_lp, end_lp[opt], days_interval[opt])
    # Haya2.textAngleEVEJAXA(ax, params, 365)
    # Ryugu.textAngleEVEJAXA(ax, params, 365)


    # fig.suptitle("Planets and Haya2 on {0:%Y-%m-%d}".format(target_date))
    # fig.savefig("planets_and_Haya2_on_{0:%Y-%m-%d}.png".format(target_date), dpi=300)

    fig.suptitle("Earth and Haya2 on Dec.3".format(target_date))
    fig.savefig("Earth_and_Haya2_20151203.png".format(target_date), dpi=300)

    print("end")


if __name__ == '__main__':
    main()
