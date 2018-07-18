#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define the classes, methods and functions for Finite
Element Method (FEM) of lowest order in tetrahedral meshes.
'''


def tetraXiEtaZeta2XYZ(eleNodes, XiEtaZetaPoints):
    ''' Map a set of points in XiEtaZeta coordinates to XYZ coordinates.

    :param ndarray eleNodes: nodal spatial coordinates of the
     tetrahedral element.
    :param ndarray XiEtaZetaPoints: set of points in XiEtaZeta coordinates.
    :return: new spatial coordinates of XiEtaZetaPoints.
    :rtype: ndarray.
    '''
    # Get number of points
    if XiEtaZetaPoints.ndim == 1:
        nPoints = 1
        # Allocate
        xyzPoints = np.zeros((3), dtype=np.float64)
        # Mapping all points
        # x-coordinates
        xyzPoints[0] = eleNodes[0][0] + \
            (eleNodes[1][0]-eleNodes[0][0])*XiEtaZetaPoints[0] + \
            (eleNodes[2][0]-eleNodes[0][0])*XiEtaZetaPoints[1] + \
            (eleNodes[3][0]-eleNodes[0][0])*XiEtaZetaPoints[2]
        # y-coordinates
        xyzPoints[1] = eleNodes[0][1] + \
            (eleNodes[1][1]-eleNodes[0][1])*XiEtaZetaPoints[0] + \
            (eleNodes[2][1]-eleNodes[0][1])*XiEtaZetaPoints[1] + \
            (eleNodes[3][1]-eleNodes[0][1])*XiEtaZetaPoints[2]
        # z-coordinates
        xyzPoints[2] = eleNodes[0][2] + \
            (eleNodes[1][2]-eleNodes[0][2])*XiEtaZetaPoints[0] + \
            (eleNodes[2][2]-eleNodes[0][2])*XiEtaZetaPoints[1] + \
            (eleNodes[3][2]-eleNodes[0][2])*XiEtaZetaPoints[2]
    else:
        nPoints = XiEtaZetaPoints.shape[0]
        # Allocate
        xyzPoints = np.zeros((nPoints, 3), dtype=np.float64)
        # Mapping all points
        # x-coordinates
        xyzPoints[:, 0] = eleNodes[0][0] + \
            ([eleNodes[1][0]-eleNodes[0][0]])*XiEtaZetaPoints[:, 0] + \
            ([eleNodes[2][0]-eleNodes[0][0]])*XiEtaZetaPoints[:, 1] + \
            ([eleNodes[3][0]-eleNodes[0][0]])*XiEtaZetaPoints[:, 2]
        # y-coordinates
        xyzPoints[:, 1] = eleNodes[0][1] + \
            ([eleNodes[1][1]-eleNodes[0][1]])*XiEtaZetaPoints[:, 0] + \
            ([eleNodes[2][1]-eleNodes[0][1]])*XiEtaZetaPoints[:, 1] + \
            ([eleNodes[3][1]-eleNodes[0][1]])*XiEtaZetaPoints[:, 2]
        # z-coordinates
        xyzPoints[:, 2] = eleNodes[0][2] + \
            ([eleNodes[1][2]-eleNodes[0][2]])*XiEtaZetaPoints[:, 0] + \
            ([eleNodes[2][2]-eleNodes[0][2]])*XiEtaZetaPoints[:, 1] + \
            ([eleNodes[3][2]-eleNodes[0][2]])*XiEtaZetaPoints[:, 2]

    return xyzPoints


def gauss_points_tetrahedron(polyOrder):
    ''' Compute the quadrature points X and the weights W for the
    integration over the unit tetrahedra whose nodes are (0,0,0),
    (1,0,0), (0,1,0) and (0,0,1).

    :param int polyOrder: degree of polynominal
    :return: quadrature Gauss points and Gauss weights.
    :rtype: ndarray.

    .. note:: References:\n
       P Keast, Moderate degree tetrahedral quadrature formulas,
       CMAME 55: 339-348 (1986).

       O.C. Zienkiewicz, The Finite Element Method, Sixth Edition.
    '''

    if polyOrder == 4:
        polyOrder = 5
    elif polyOrder > 14:
        polyOrder = 14

    def one():
        ''' 1 gauss point.
        '''
        w = 1.0
        [X, W] = s4(w)

        return (X, W)

    def two():
        ''' 4 gauss points.
        '''
        w = 1.0/4.0
        a = 0.1381966011250105151795413165634361
        [X, W] = s31(a, w)

        return (X, W)

    def three():
        ''' 8 gauss points.
        '''
        w = np.float64(0.1385279665118621423236176983756412)
        a = np.float64(0.3280546967114266473358058199811974)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.1114720334881378576763823016243588)
        a = np.float64(0.1069522739329306827717020415706165)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=0)

        return (X, W)

    def four():
        ''' 14 gauss points.
        '''
        w = np.float64(0.0734930431163619493435869458636788)
        a = np.float64(0.0927352503108912262865589206603214)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.1126879257180158503650149284763889)
        a = np.float64(0.3108859192633006097581474949404033)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=0)
        W = np.concatenate((W, weights), axis=0)

        w = np.float64(0.0425460207770814668609320837732882)
        a = np.float64(0.0455037041256496500000000000000000)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=0)
        W = np.concatenate((W, weights), axis=0)

        return (X, W)

    def five():
        ''' 14 gauss points.
        '''
        w = np.float64(0.1126879257180158507991856523332863)
        a = np.float64(0.3108859192633006097973457337634578)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.0734930431163619495437102054863275)
        a = np.float64(0.0927352503108912264023239137370306)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0425460207770814664380694281202574)
        a = np.float64(0.0455037041256496494918805262793394)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def six():
        ''' 24 gauss points.
        '''
        w = np.float64(.0399227502581674920996906275574800)
        a = np.float64(.2146028712591520292888392193862850)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(.0100772110553206429480132374459369)
        a = np.float64(.0406739585346113531155794489564101)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0553571815436547220951532778537260)
        a = np.float64(0.3223378901422755103439944707624921)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0482142857142857142857142857142857)
        a = np.float64(0.0636610018750175252992355276057270)
        b = np.float64(0.6030056647916491413674311390609397)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def seven():
        ''' 36 gauss points.
        '''
        w = np.float64(0.0061834158394585176827283275896323)
        a = np.float64(0.0406107071929452723515318677627212)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.0785146502738723588282424885149114)
        a = np.float64(0.1787522026964984761546314943983834)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0447395776143224792777362432057442)
        a = np.float64(0.3249495905373373335715573286644841)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0121651445922912935604045907106762)
        a = np.float64(0.1340777379721611918326213913378565)
        b = np.float64(0.7270125070093171000000000000000000)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0280223074984909211766930561858945)
        a = np.float64(0.0560275404597284769655799958528421)
        b = np.float64(0.3265740998664049580757011076659178)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def eight():
        ''' 46 gauss points.
        '''
        w = np.float64(0.0063971477799023213214514203351730)
        a = np.float64(0.0396754230703899012650713295393895)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.0401904480209661724881611584798178)
        a = np.float64(0.3144878006980963137841605626971483)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0243079755047703211748691087719226)
        a = np.float64(0.1019866930627033000000000000000000)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0548588924136974404669241239903914)
        a = np.float64(0.1842036969491915122759464173489092)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0357196122340991824649509689966176)
        a = np.float64(0.0634362877545398924051412387018983)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0071831906978525394094511052198038)
        a = np.float64(0.0216901620677280048026624826249302)
        b = np.float64(0.7199319220394659358894349533527348)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0163721819453191175409381397561191)
        a = np.float64(0.2044800806367957142413355748727453)
        b = np.float64(0.5805771901288092241753981713906204)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def nine():
        ''' 61 gauss points.
        '''
        w = np.float64(0.0564266931795062065887150432761254)
        [points, weights] = s4(w)
        X = np.vstack(points)
        W = weights

        w = np.float64(0.0033410950747134804029997443047177)
        a = np.float64(0.0340221770010448664654037088787676)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0301137547687737639073142384315749)
        a = np.float64(0.3227703335338005253913766832549640)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0064909609200615346357621168945686)
        a = np.float64(0.0604570774257749300000000000000000)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0098092858682545864319687425925550)
        a = np.float64(0.4553629909472082118003081504416430)
        b = np.float64(0.0056831773653301799061001601457447)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0281191538233654725516326174252926)
        a = np.float64(0.1195022553938258009779737046961144)
        b = np.float64(0.4631168324784899409762244936577296)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0078945869083315007683414920096088)
        a = np.float64(0.0280219557834011581550575066541237)
        b = np.float64(0.7252060768398674887385659542848099)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0194928120472399967169721944892460)
        a = np.float64(0.1748330320115746157853246459722452)
        b = np.float64(0.6166825717812564045706830909795407)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def ten():
        ''' 81 gauss points.
        '''
        w = np.float64(0.0473997735560207383847388211780511)
        [points, weights] = s4(w)
        X = np.vstack(points)
        W = weights

        w = np.float64(0.0269370599922686998027641610048821)
        a = np.float64(0.3122500686951886477298083186868275)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0098691597167933832345577354301731)
        a = np.float64(0.1143096538573461505873711976536504)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0003619443443392536242398783848085)
        a = np.float64(0.0061380088247907478475937132484154)
        b = np.float64(0.9429887673452048661976305869182508)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0101358716797557927885164701150168)
        a = np.float64(0.0327794682164426707747210203323242)
        b = np.float64(0.3401847940871076327889879249496713)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0113938812201952316236209348807143)
        a = np.float64(0.4104307392189654942878978442515117)
        b = np.float64(0.1654860256196110516044901244445264)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0065761472770359041674557402004507)
        a = np.float64(0.0324852815648230478355149399784262)
        b = np.float64(0.1338521522120095130978284359645666)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0257397319804560712790360122596547)
        a = np.float64(0.1210501811455894259938950015950505)
        b = np.float64(0.4771903799042803505441064082969072)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0129070357988619906392954302494990)
        a = np.float64(0.1749793421839390242849492265283104)
        b = np.float64(0.6280718454753660106932760722179097)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def eleven():
        ''' 109 gauss points.
        '''
        w = np.float64(0.0394321080286588635073303344912044)
        [points, weights] = s4(w)
        X = np.vstack(points)
        W = weights

        w = np.float64(0.0156621262272791131500885627687651)
        a = np.float64(0.1214913677765337944977023099080722)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0033321723749014081444092361540149)
        a = np.float64(0.0323162591510728963539544520895810)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0140260774074897474374913609976924)
        a = np.float64(0.3249261497886067978128419024144220)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0010859075293324663068220983772355)
        a = np.float64(0.0041483569716600120000000000000100)
        b = np.float64(0.5982659967901863502054538427761778)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0202359604306631789111165731654084)
        a = np.float64(0.2246246106763771414144751511649864)
        b = np.float64(0.4736622878323495714083696692020524)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0117902148721258635368493804677018)
        a = np.float64(0.0519050877725656967442272164426589)
        b = np.float64(0.5631447779082798987371019763030571)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0076903149825212959011315780207389)
        a = np.float64(0.1349301312162402042237591723429930)
        b = np.float64(0.7083588307858189538569950051271300)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0044373057034592039047307260214396)
        a = np.float64(0.0251911921082524729200511850653055)
        b = np.float64(0.7837195073400773754305740342999090)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0114295484671840404107705525985940)
        a = np.float64(0.3653187797817336139693319800988672)
        b = np.float64(0.1346039083168658000000000000000100)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0061856401712178114128192550838953)
        a = np.float64(0.5229075395099384729652169275860292)
        b = np.float64(0.1407536305436959018425391394912785)
        c = np.float64(0.0097624381964526155082922803899778)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def twelve():
        ''' 140 gauss points.
        '''
        w = np.float64(0.0127676377009707415020377859651250)
        a = np.float64(0.1152997443514801453045572073891591)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.0161211042379092682185815448957576)
        a = np.float64(0.2023362822405909000000000000000100)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0003716126985784422000425581898608)
        a = np.float64(0.0117175979576199515124790675483140)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0197174417866854576395533090381887)
        a = np.float64(0.3133064413678010672776027996445893)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0025713909308627183621823475944855)
        a = np.float64(0.2500057301155837000000000000000100)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0038172478705105759057531841278333)
        a = np.float64(0.0209954743507580066902018252705902)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0120872270776631131786031841931461)
        a = np.float64(0.1517740182474501000000000000000100)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0031058611584347334343168814992962)
        a = np.float64(0.0244197787434353647831400090476166)
        b = np.float64(0.8483292846978728506452088674348157)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0054595313364710306691274212676944)
        a = np.float64(0.2562070985320183089638201070856221)
        b = np.float64(0.4824873738738488478028928967297354)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0021428997484969975066685209365595)
        a = np.float64(0.0167903209796029906147179602885794)
        b = np.float64(0.6947719423657559269594985098841772)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0055246714672578296224493009816508)
        a = np.float64(0.1261608211398720423997070384689592)
        b = np.float64(0.7254104893029481189748595052126338)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0085369566944991804298517783667220)
        a = np.float64(0.4314351745263798472167069506637196)
        b = np.float64(0.1127219398928524152095997721100754)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0115101778483233069733364412340329)
        a = np.float64(0.5016700624625056974751550716847613)
        b = np.float64(0.2724718028695223917835104675306045)
        c = np.float64(0.0720743288072989146501594845633582)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0052038786528856136039679242125245)
        a = np.float64(0.2616448545378187456694550500639680)
        b = np.float64(0.0862922919470617319174235194435249)
        c = np.float64(0.0205654106558761383006248976271090)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def thirteen():
        ''' 171 gauss points.
        '''
        w = np.float64(0.0150136877730831467506297063161598)
        [points, weights] = s4(w)
        X = np.vstack(points)
        W = weights

        w = np.float64(0.0182252092801734253237906894149010)
        a = np.float64(0.1552160935190895031411578433570474)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0070061092177414642403851869392631)
        a = np.float64(0.3301226633396736002443319259519678)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0164235497439495482954057310790553)
        a = np.float64(0.1668064038938624992893778260114423)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0051206100963605970726259694970217)
        a = np.float64(0.0249237885477736177970140037486009)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0111966986529049163438203208635196)
        a = np.float64(0.0971976299157510014307224371624082)
        [points, weights] = s22(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0156191497333799540095381130243197)
        a = np.float64(0.2478592901573625669274691062082793)
        b = np.float64(0.4336532423568514471872606143476738)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0024844230133164744190405677633847)
        a = np.float64(0.0222315960818670029087952186089293)
        b = np.float64(0.8369003204037340051450948659569859)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0016385985348182389384452530944075)
        a = np.float64(0.1072786933130534104915045963958480)
        b = np.float64(0.7749803059750018075658787727417929)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0059030304401249219717191465553587)
        a = np.float64(0.1981768438839898114233184058214276)
        b = np.float64(0.5875693057822053025917201790359592)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0110220824582180524044509798920153)
        a = np.float64(0.0691792434773793164773253434746550)
        b = np.float64(0.6042000666600664470793526487111530)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0004064518399641782258515551275585)
        a = np.float64(0.0231147194719331600000000000000100)
        b = np.float64(0.9308757927924442486492022888288831)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0026879699729685420974578192665173)
        a = np.float64(0.1178892875101960892229011747064425)
        b = np.float64(0.1165153642254072000000000000000100)
        c = np.float64(0.0420240011255154209567663430372000)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0019795048055267119053189467551074)
        a = np.float64(0.6770327986022842635503222132674659)
        b = np.float64(0.0461653760246197108345804112217608)
        c = np.float64(0.0008443403189050397572989969213590)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0054463191814257912094318704010867)
        a = np.float64(0.4848900886736331220108009415479083)
        b = np.float64(0.3588829429552020157242364690942109)
        c = np.float64(0.1381828349176287299695508090791236)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def fourteen():
        ''' 236 gauss points.
        '''
        w = np.float64(0.0040651136652707670436208836835636)
        a = np.float64(0.3272533625238485639093096692685289)
        [points, weights] = s31(a, w)
        X = points
        W = weights

        w = np.float64(0.0022145385334455781437599569500071)
        a = np.float64(0.0447613044666850808837942096478842)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0058134382678884505495373338821455)
        a = np.float64(0.0861403311024363536537208740298857)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0196255433858357215975623333961715)
        a = np.float64(0.2087626425004322968265357083976176)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0003875737905908214364538721248394)
        a = np.float64(0.0141049738029209600635879152102928)
        [points, weights] = s31(a, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0116429719721770369855213401005552)
        a = np.float64(0.1021653241807768123476692526982584)
        b = np.float64(0.5739463675943338202814002893460107)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0052890429882817131317736883052856)
        a = np.float64(0.4075700516600107157213295651301783)
        b = np.float64(0.0922278701390201300000000000000000)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0018310854163600559376697823488069)
        a = np.float64(0.0156640007402803585557586709578084)
        b = np.float64(0.7012810959589440327139967673208426)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0082496473772146452067449669173660)
        a = np.float64(0.2254963562525029053780724154201103)
        b = np.float64(0.4769063974420887115860583354107011)
        [points, weights] = s211(a, b, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0030099245347082451376888748208987)
        a = np.float64(0.3905984281281458000000000000000000)
        b = np.float64(0.2013590544123922168123077327235092)
        c = np.float64(0.0161122880710300298578026931548371)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0008047165617367534636261808760312)
        a = np.float64(0.1061350679989021455556139029848079)
        b = np.float64(0.0327358186817269284944004077912660)
        c = np.float64(0.0035979076537271666907971523385925)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0029850412588493071187655692883922)
        a = np.float64(0.5636383731697743896896816630648502)
        b = np.float64(0.2302920722300657454502526874135652)
        c = np.float64(0.1907199341743551862712487790637898)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0056896002418760766963361477811973)
        a = np.float64(0.3676255095325860844092206775991167)
        b = np.float64(0.2078851380230044950717102125250735)
        c = np.float64(0.3312104885193449000000000000000000)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0041590865878545715670013980182613)
        a = np.float64(0.7192323689817295295023401840796991)
        b = np.float64(0.1763279118019329762157993033636973)
        c = np.float64(0.0207602362571310090754973440611644)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0007282389204572724356136429745654)
        a = np.float64(0.5278249952152987298409240075817276)
        b = np.float64(0.4372890892203418165526238760841918)
        c = np.float64(0.0092201651856641949463177554949220)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        w = np.float64(0.0054326500769958248216242340651926)
        a = np.float64(0.5483674544948190728994910505607746)
        b = np.float64(0.3447815506171641228703671870920331)
        c = np.float64(0.0867217283322215394629438740085828)
        [points, weights] = s1111(a, b, c, w)
        X = np.concatenate((X, points), axis=1)
        W = np.concatenate((W, weights), axis=1)

        return (X, W)

    def s4(w):
        ''' The first star contains only this one point
        '''
        X = np.array([1.0/4.0,  1.0/4.0,  1.0/4.0], dtype=np.float64)
        W = np.array([w], dtype=np.float64)

        return (X, W)

    def s31(a, w):
        ''' First star: Compute the barycentric coordinates, which
        contain 4 dimensions. The points are obtained by taking all
        the unique 3 dimensional permutations from the barycentric coordinates.
        '''
        baryc = [a, a, a, (1.0-3.0*a)]
        temp = np.array(list(itertools.permutations(baryc)), dtype=np.float64)
        [dummy, _, _] = findUniqueRows(temp, return_index=True,
                                       return_inverse=True)
        X = dummy[:, 0:3]
        X = X.transpose()
        W = w * np.ones(X.shape[1])

        return (X, W)

    def s22(a, w):
        ''' Second star: Compute the barycentric coordinates, which
        contain 4 dimensions. The points are obtained by taking all
        the unique 3 dimensional permutations from the barycentric coordinates.
        '''
        baryc = [a, a, 0.5-a, 0.5-a]
        temp = np.array(list(itertools.permutations(baryc)), dtype=np.float64)
        [dummy, _, _] = findUniqueRows(temp, return_index=True,
                                       return_inverse=True)
        X = dummy[:, 0:3]
        X = X.transpose()
        W = w * np.ones(X.shape[1])

        return (X, W)

    def s211(a, b, w):
        ''' Second star: Compute the barycentric coordinates, which
        contain 4 dimensions. The points are obtained by taking all
        the unique 3 dimensional permutations from the barycentric coordinates.
        '''
        baryc = [a, a, b, (1.0-2.0*a-b)]
        temp = np.array(list(itertools.permutations(baryc)), dtype=np.float64)
        [dummy, _, _] = findUniqueRows(temp, return_index=True,
                                       return_inverse=True)
        X = dummy[:, 0:3]
        X = X.transpose()
        W = w * np.ones(X.shape[1])

        return (X, W)

    def s1111(a, b, c, w):
        ''' Fourth star: Compute the barycentric coordinates, which
        contain 4 dimensions. The points are obtained by taking all
        the unique 3 dimensional permutations from the barycentric coordinates.
        '''
        baryc = [a, b, c, (1.0-a-b-c)]
        temp = np.array(list(itertools.permutations(baryc)), dtype=np.float64)
        [dummy, _, _] = findUniqueRows(temp, return_index=True,
                                       return_inverse=True)
        X = dummy[:, 0:3]
        X = X.transpose()
        W = w * np.ones(X.shape[1])

        return (X, W)

    # Options for Gauss points computation (switch case)
    options = {1: one,
               2: two,
               3: three,
               4: four,
               5: five,
               6: six,
               7: seven,
               8: eight,
               9: nine,
               10: ten,
               11: eleven,
               12: twelve,
               13: thirteen,
               14: fourteen}

    [X, W] = options[polyOrder]()

    X = np.transpose(X)
    W = np.transpose(W)

    return (X, W)


def unitary_test():
    ''' Unitary test for fem.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    unitary_test()
else:
    # Standard module import
    import itertools
    import numpy as np
    # PETGEM module import
    from petgem.efem.vectorMatrixFunctions import findUniqueRows
