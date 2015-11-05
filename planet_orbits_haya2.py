#python
# -*- coding:utf-8 -*-
# Time-stamp: <Thu Nov 05 17:19:13 JST 2015>

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines


### USER PARAMETER ###
DRAW_INNER = True  # plot inner of Mars orbit / plot all planets Orbit
target_date = datetime.date(2015, 5, 15) # plot position on target_date


theta, phi = 0, 0
mag = 1.                        # magnification
eve_x, eve_y = 0, 0
DRAW_INNER = True


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

# global variables
fig = None
ax  = None

class Planet:
    def __init__(self, name):
        self.name = name
        self.xl = []
        self.yl = []
        self.zl = []
        self.rl = []
        self.r_xyl = []

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


    def calc(self, jd):
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
        self.E = solveE(self.M, self.e)

        x_prime = self.a * (math.cos(self.E) - self.e)
        y_prime = self.b * math.sin(self.E)
        r_prime = math.sqrt(x_prime ** 2 + y_prime ** 2)

        mat = getMatrix(self.I, self.Omega, self.omega)

        self.x = mat[0][0] * x_prime + mat[0][1] * y_prime
        self.y = mat[1][0] * x_prime + mat[1][1] * y_prime
        self.z = mat[2][0] * x_prime + mat[2][1] * y_prime

        self.r = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        self.r_xy = math.sqrt(self.x*self.x + self.y*self.y)

        # plot x, y, z pos: px, py, pz
        self.px, self.py, self.pz = convertCood(self.x, self.y, self.z)

        self.xl.append(self.px)
        self.yl.append(self.py)
        self.zl.append(self.pz)
        self.rl.append(self.r)

        self.r_xyl.append(self.r_xy)

    def plot(self):
        '''
        plot line of target planet (for planet orbit)
        '''
        plt.plot(self.xl, self.yl, '-', lw=0.05,
                 color = {'Mercury': 'b',
                          'Venus'  : '#ffd700',
                          'Earth'  : 'g',
                          'Mars'   : 'r',
                          'Jupiter': '#8b4512',
                          'Saturn' : '#deb887',
                          'Uranus' : '#40e0d0',
                          'Neptune': '#00bfff',
                          'Pluto'  : 'k'
                         }[self.name]
        )

    def qQ(self):
        print("-------------------")
        print(self.name)
        print(" perihelion distance[q] : {:.3f}\n aperihelion distance[Q]: {:.3f}".format(min(self.rl), max(self.rl)))
        print("-------------------")

    def plot2(self):
        '''
        plot point of target planet on target date
        '''
        plt.plot(self.xl, self.yl, '*', ms=12,
                 color = {'Mercury': 'b',
                          'Venus'  : '#ffd700',
                          'Earth'  : 'g',
                          'Mars'   : 'r',
                          'Jupiter': '#8b4513',
                          'Saturn' : '#deb887',
                          'Uranus' : '#40e0d0',
                          'Neptune': '#00bfff',
                          'Pluto'  : 'k'
                         }[self.name]
        )

    def plot3(self, dl):
        '''
        caption text of planets
        '''
        for i in range(len(self.xl)):
            plt.plot(self.xl[i], self.yl[i], '*', ms=12,
                     color = {'Mercury': 'b',
                              'Venus'  : '#ffd700',
                              'Earth'  : 'g',
                              'Mars'   : 'r',
                              'Jupiter': '#8b4513',
                              'Saturn' : '#deb887',
                              'Uranus' : '#40e0d0',
                              'Neptune': '#00bfff',
                              'Pluto'  : 'k'
                          }[self.name]
            )
            plt.text(self.xl[i], self.yl[i], "${0:%m/%d}^{{\mathrm{{'}}{0:%y}}}$".format(dl[i]),
                     fontsize=6)

    def plotBase(self):
        ''''''
        plt.plot(self.xl, self.yl, 'D',
                 color = {'Mercury': 'b',
                          'Earth'  : '#00fa9a',
                          'Pluto'  : '#ff1493'
                         }[self.name]
        )


    def plotVE(self):
        '''
        plot line between Earth and Sun
        '''
        plt.plot(self.xl, self.yl, 'x', color='g')

        line = lines.Line2D([0,self.x], [0, self.y])
        ax.add_line(line)

    def distance(self):
        print("*********")
        print(self.name)
        print(" distance from Sun: {:.3f}".format(self.r))
        print("*********")

    def distanceXY(self):
        print("{:s} distance from Sun(x-y): {:.3f}".format(self.name, distance2point(np.array([self.px, self.py]), np.array([0, 0]))))

    def angleEVE(self):
        '''
        angle Earth Vernal Equinox day position - Sun position and Planet position - Sun position
        '''
        global eve_x, eve_y
        a = np.array([self.x, self.y])
        b = np.array([eve_x, eve_y])

        ang = angle2vector(a, b) # ang[radian]
        ang_deg = math.degrees(ang)

        print("{:s} angle(x-y): {:.3f}".format(self.name, ang_deg))
        plt.text(self.px, self.py, "{:.1f}".format(ang_deg))

class HAYA2:
    def __init__(self):
        self.lst = []
        h2txt = open("./haya2_orbit_jaxa.txt","r")
        h2list = [x.strip() for x in h2txt.readlines()][51:]
        for line in h2list:
            _d = {}
            # _d['date'] = line[0:10].replace("/", "") # target date
            _date = datetime.datetime.strptime(line[0:19], "%Y/%m/%d.%H:%M:%S")
            _d['date'] = datetime.date(_date.year, _date.month, _date.day) # date type
            _d['lp']   = int(line[22:26])          # L+ [days]
            _d['x']  = float(line[29:38])        # X pos. [au]
            _d['y']  = float(line[39:48])        # Y pos. [au]
            _d['z']  = float(line[49:58])        # Z pos. [au]
            _d['ex'] = float(line[59:68])
            _d['ey'] = float(line[69:78])
            _d['ez'] = float(line[79:88])
            _d['rs'] = float(line[122:129])      # distance of Sun-Haya2 [10**4 km]
            _d['re'] = float(line[133:140])      # distance of Earth-Haya2 [10**4 km]
            _d['ra'] = float(line[144:151])      # distance of 1999JU3-Haya2 [10**4 km]
            _d['vs']  = float(line[154:159])     # velocity of Haya2 on Sun [km/sec]
            _d['ve']  = float(line[162:167])     # velocity of Haya2 on Earth [km/sec]
            _d['alpha'] = float(line[169:176])   # ra [deg]
            _d['delta'] = float(line[179:185])   # dec [deg]
            _d['Dflt']  = float(line[186:194])   # distance of fling [10**4 km]

            _d['px'], _d['py'], _d['pz'] = convertCood(_d['x'], _d['y'], _d['z'])

            self.lst.append(_d)

def plotHaya2(day):
    haya2 = HAYA2()
    for h in haya2.lst[0:365]:
        plt.plot(h['px'], h['py'], "k.", ms=1)
    for h in haya2.lst[0:365:30]:
        plt.plot(h['px'], h['py'], "c.")
        plt.text(h['px'], h['py'], "${date:%m/%d}^{{\mathrm{{'}}{date:%y}}}$".format(**h),
                 ha='center', va='bottom', fontsize=6, color='blue')

class JU3:
    def __init__(self):
        self.lst = []
        h2txt = open("./haya2_orbit_jaxa.txt","r")
        h2list = [x.strip() for x in h2txt.readlines()][51:]
        for line in h2list:
            _d = {}
            # _d['date'] = line[0:10].replace("/", "") # target date
            _date = datetime.datetime.strptime(line[0:19], "%Y/%m/%d.%H:%M:%S")
            _d['date'] = datetime.date(_date.year, _date.month, _date.day) # date type
            _d['lp']   = int(line[22:26])          # L+ [days]
            _d['x']  = float(line[89:98])        # X pos. [au]
            _d['y']  = float(line[99:108])        # Y pos. [au]
            _d['z']  = float(line[109:118])        # Z pos. [au]
            _d['ra'] = float(line[144:151])      # distance of 1999JU3-Haya2 [10**4 km]

            _d['px'], _d['py'], _d['pz'] = convertCood(_d['x'], _d['y'], _d['z'])

            self.lst.append(_d)


def plot1999JU3(day):
    ju = JU3()
    for p in ju.lst[0:365]:
        plt.plot(p['px'], p['py'], "k.", ms=1)
    for p in ju.lst[0:365:30]:
        plt.plot(p['px'], p['py'], "cH")
        plt.text(p['px'], p['py'], "${date:%m/%d}^{{\mathrm{{'}}{date:%y}}}$".format(**p),
                 ha='center', va='bottom', fontsize=3, color='green')


def convertCood(x, y, z):
    global theta, phi
    # rotation theta around x-axis
    _theta = math.radians(theta)
    x2 = x
    y2 = y * math.cos(_theta) - z * math.sin(_theta)
    z2 = y * math.sin(_theta) + z * math.cos(_theta)

    # rotation phi around y-axis
    _phi = math.radians(phi)
    x3 = x2 * math.cos(_phi) + z2 * math.sin(_phi)
    y3 = y2
    z3 = z2 * math.cos(_phi) - x2 * math.sin(_phi)

    return [x3, y3, z3]


def getMatrix(I, Omega, omega):
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


def solveE(M, e):
    count = 1
    E1 = M
    while True:
        # E2 = M + math.degrees(e * math.sin(math.radians(E1)))
        delta_E = (M - E1 + e * math.sin(E1)) / (1 - e * math.cos(E1))
        E2 = E1 + delta_E
        E1 = E2

        if math.fabs(delta_E) < 0.00000001:
            # print(count)
            break
        if count % 1000 == 0:
            print("count of solveE: {}".format(count))
        if count > 10000: exit(0)
        count += 1
    return E2


def toJD(da):
    y = da.year
    m = da.month

    if m < 3:
        m = m + 12
        y = y - 1

    jd_year = int(365.25 * y) - int(y/100) + int(y/400) + 1721088.5
    jd_day = int(30.59 * (m -2)) + int(da.day)
    jd = jd_year + jd_day
    return jd

def drawOrbit():
    global fig, ax
    begin_date = datetime.date(1990,1,1)
    end_date = datetime.date(2030,1,1)
    days_interval = 10                   # interval days

    fig = plt.figure(figsize=(10,10))
    ax  = fig.add_subplot(111)

    Mercury = Planet("Mercury")
    Venus = Planet("Venus")
    Earth = Planet("Earth")
    Mars = Planet("Mars")
    Jupiter = Planet("Jupiter")
    Saturn = Planet("Saturn")
    Uranus = Planet("Uranus")
    Neptune = Planet("Neptune")
    Pluto = Planet("Pluto")


    count = 0
    while True:
        day = begin_date + datetime.timedelta(days = count * days_interval)
        jd = toJD(day) - 2451545.0 # J2000


        Mercury.calc(jd)
        Venus.calc(jd)
        Earth.calc(jd)
        Mars.calc(jd)
        Jupiter.calc(jd)
        Saturn.calc(jd)
        Uranus.calc(jd)
        Neptune.calc(jd)
        Pluto.calc(jd)

        if (end_date - day).days < 0: break
        count += 1

    Mercury.plot()
    Venus.plot()
    Earth.plot()
    Mars.plot()
    Jupiter.plot()
    Saturn.plot()
    Uranus.plot()
    Neptune.plot()
    Pluto.plot()

    plt.axis('equal')
    global DRAW_INNER
    # DRAW_INNER = False
    # DRAW_INNER = True
    if DRAW_INNER:
        # plt.xlim(min(Jupiter.xl), max(Jupiter.xl))
        # plt.ylim(min(Jupiter.yl), max(Jupiter.yl))
        plt.axis([-2, 2, -2, 2])
        # plt.xlim(min(-2), max(2))
        # plt.ylim(min(-2), max(Mars.yl))

    Mercury.qQ()
    Venus.qQ()
    Earth.qQ()
    Mars.qQ()
    Jupiter.qQ()
    Saturn.qQ()
    Uranus.qQ()
    Neptune.qQ()
    Pluto.qQ()


def plotPlanetsMulti():
    begin_date = datetime.date(2015,11,3)
    end_date = datetime.date(2016,1,3)
    days_interval = 30                   # interval days

    sui = Planet("Mercury")
    kin = Planet("Venus")
    chi = Planet("Earth")
    ka = Planet("Mars")
    moku = Planet("Jupiter")
    do = Planet("Saturn")
    ten = Planet("Uranus")
    kai = Planet("Neptune")
    mei = Planet("Pluto")

    daylist = []
    count = 0
    while True:
        day = begin_date + datetime.timedelta(days = count * days_interval)
        daylist.append(day)
        jd = toJD(day) - 2451545.0 # J2000


        sui.calc(jd)
        kin.calc(jd)
        chi.calc(jd)
        ka.calc(jd)
        moku.calc(jd)
        do.calc(jd)
        ten.calc(jd)
        kai.calc(jd)
        mei.calc(jd)

        if (end_date - day).days < 0: break
        count += 1

    sui.plot3(daylist)
    kin.plot3(daylist)
    chi.plot3(daylist)
    ka.plot3(daylist)
    moku.plot3(daylist)
    do.plot3(daylist)
    ten.plot3(daylist)
    kai.plot3(daylist)
    mei.plot3(daylist)

def plotEarthMulti():
    begin_date = datetime.date(2014,12,3)
    end_date = datetime.date(2015,12,3)
    days_interval = 30                   # interval days

    chi = Planet("Earth")

    daylist = []
    count = 0
    while True:
        day = begin_date + datetime.timedelta(days = count * days_interval)
        daylist.append(day)
        jd = toJD(day) - 2451545.0 # J2000


        chi.calc(jd)

        if (end_date - day).days < 0: break
        count += 1

    chi.plot3(daylist)


def plotPlanets(day):
    mer = Planet("Mercury")
    ven = Planet("Venus")
    ear = Planet("Earth")
    mar = Planet("Mars")
    jup = Planet("Jupiter")
    sat = Planet("Saturn")
    ura = Planet("Uranus")
    nep = Planet("Neptune")
    plu = Planet("Pluto")

    jd = toJD(day) - 2451545.0 # J2000 not used

    mer.calc(jd)
    ven.calc(jd)
    ear.calc(jd)
    mar.calc(jd)
    jup.calc(jd)
    sat.calc(jd)
    ura.calc(jd)
    nep.calc(jd)
    plu.calc(jd)

    mer.plot2()
    ven.plot2()
    ear.plot2()
    mar.plot2()
    jup.plot2()
    sat.plot2()
    ura.plot2()
    nep.plot2()
    plu.plot2()


    print("-----------------------")

    mer.distanceXY()
    ven.distanceXY()
    ear.distanceXY()
    mar.distanceXY()
    jup.distanceXY()
    sat.distanceXY()
    ura.distanceXY()
    nep.distanceXY()
    plu.distanceXY()

    # nep.distance()
    # plu.distance()

    print("-----------------------")

    mer.angleEVE()
    ven.angleEVE()
    ear.angleEVE()
    mar.angleEVE()
    jup.angleEVE()
    sat.angleEVE()
    ura.angleEVE()
    nep.angleEVE()
    plu.angleEVE()

    # print("haya2")
    # print(ear.x)
    # print(ear.y)
    # print(ear.z)


def plotVEquinox(day):
    earth = Planet("Earth")

    jd = toJD(day) - 2451545.0 # J2000 not used
    earth.calc(jd)

    earth.plotVE()

def distance2point(a, b):       # a,b: numpy array([x1,y1], [x2,y2])
    global mag
    u = b - a
    return np.linalg.norm(u * mag)

def angle2vector(a, b):           # a,b: vector (numpy)
    cos_ang = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ang = math.acos(cos_ang)
    return ang                  # radian

def drawBase():
    global eve_x, eve_y
    eve = Planet("Earth")       # Earth on Vernal Equinox Day
    jdeve = toJD(datetime.date(2014, 3, 21)) - 2451545.0 # J2000 not used
    eve.calc(jdeve)
    eve.plotBase()

    eve_x, eve_y = eve.px, eve.py # eve position on plot
    print(eve_x)
    print(eve_y)

def main():
    global theta, phi, mag, DRAW_INNER
    theta, phi = 6, -8
    mag = 0.245 / 1.

    drawOrbit()                 # Orbit draw

    plt.plot(0, 0, "ro")
    drawBase()

    # plotPlanets(target_date) # plot planet on target day(arg[datetime type])
    # plotPlanetsMulti()
    plotEarthMulti()

    # plot1999JU3(target_date)

    plotHaya2(target_date)

    plt.title(target_date.strftime("Inner Planets + Haya2 every 30 days"), fontsize=20)
    plt.xlabel("$x[\mathrm{au}]$", fontsize=20)
    plt.ylabel("$y[\mathrm{au}]$", fontsize=20)
    # plt.savefig(target_date.strftime("%Y-%m-%d + Haya2") + '.png', format='png', dpi=300)
    plt.savefig(target_date.strftime("Inner Planets + Haya2 every 30 days") + '.png', format='png', dpi=300)
    print("end")


if __name__ == '__main__':
    main()
