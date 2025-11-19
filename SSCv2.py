import streamlit as st
from numpy import arange
import pandas as pd
import math
from collections import Counter

md1 = ['S-AG 38',
       'S-LAG 38', 
       'S-AG 65', 
       'S-LAG 65', 
       'S-DG 38', 
       'S-LT 38', 
       'S-DG 65', 
       'S-LT 65', 
       'S-TIF 65', 
       'S-LTIF 65',
       'S-AG 100']
md2 = ['S-AS 65',
       'S-AS 99',
       'S-LTIFAS 65',
       'S-LTIFAS 99',
       'S-TIFAS 65',
       'S-TIFAS 99']
md4 = [1,2,3,4,5,6]
st.image('https://uploads-ssl.webflow.com/61dd080b3d2f9fec43b08948/61e5d398a2373f0282831465_InfinityDrainLogo_blk.png', caption = 'Infinite Possibilities')
st.title("Site Sizable Calculator")
outdoor = False
turns = 0
pool = False
turn = False
stainless = False
comp1 = []
comp2 = []
aa1 = []
col11, col12, col13 = st.columns(3)
p1 = ''
leng0 = ''
leng1 = ''
leng2 = ''
leng3 = ''
leng4= ''
leng5 = ''
straight = False
outlet = [False,0]
outTurn = []

with col11:
    ind = st.radio("Select Application:",
                ("Indoor", "Outdoor"), key = 'outdoor')
    if ind == 'Outdoor':
        outdoor = True
    with col13:
        ai = st.checkbox("Pool Surround or Turns", value = False, key = 'ai')
        if ai:
            p1 = st.radio('',
                        ('Pool Surround', 'Turns'), key = 'pool', label_visibility='collapsed')
            if p1 == 'Pool Surround':
                pool = True
            else:
                turn = True
                turns = st.selectbox('Select number of turns:', md4)
        otltop = st.checkbox('Specify outlets', value = False, key = 'oltop')
        if otltop:
            outlet[0] = otltop
with col12:
    chMAT = st.radio("Select Channel Material:",
                    ("PVC Channel", "Stainless Steel Channel"), key = 'chann')



if chMAT == "PVC Channel":
    mod = st.selectbox("Select Model:", md1, key = 'mod')
elif chMAT == "Stainless Steel Channel":
    mod = st.selectbox("Select Model:", md2, key = 'mod1')
    stainless = True
if turn == True:
    if turns == 1:
        colt0, colt1 = st.columns(2)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', max_value=2400, key = 'l1')
        if outlet[0] == True:
            colo0, colo1 = st.columns(2)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 1, max_value = 480, key = 'ot0')
            with colo1:
                ot1 = st.number_input('', min_value = 1, max_value = 480, label_visibility='hidden',  key = 'ot1')
            outTurn.append(ot0)
            outTurn.append(ot1)
    elif turns == 2:
        colt0, colt1, colt2 = st.columns(3)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l1', max_value=2400)
        with colt2:
            leng2 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l2', max_value=2400)
        if outlet[0] == True:
            colo0, colo1, colo2 = st.columns(3)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
            with colo1:
                ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
            with colo2:
                ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot2')
            outTurn.append(ot0)
            outTurn.append(ot1)
            outTurn.append(ot2)
    elif turns == 3:
        colt0, colt1 = st.columns(2)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
            leng2 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l2', max_value=2400)
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l1', max_value=2400)
            leng3 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l3', max_value=2400)
        if outlet[0] == True:
            colo0, colo1 = st.columns(2)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
                ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot2')
            with colo1:
                ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
                ot3 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot3')
            outTurn.append(ot0)
            outTurn.append(ot1)
            outTurn.append(ot2)
            outTurn.append(ot3)
    elif turns == 4:
        colt0, colt1 = st.columns(2)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
            leng2 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l2', max_value=2400)
            leng4 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l4', max_value=2400)
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l1', max_value=2400)  
            leng3 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l3', max_value=2400)
        if outlet[0] == True:
            colo0, colo1 = st.columns(2)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
                ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot2')
                ot4 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot4')
            with colo1:
                ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
                ot3 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot3')
            outTurn.append(ot0)
            outTurn.append(ot1)
            outTurn.append(ot2)
            outTurn.append(ot3)
            outTurn.append(ot4)
    elif turns == 5:
        colt0, colt1 = st.columns(2)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
            leng2 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l2', max_value=2400)
            leng4 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l4', max_value=2400)
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l1', max_value=2400)
            leng3 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l3', max_value=2400)
            leng5 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l5', max_value=2400)
        if outlet[0] == True:
            colo0, colo1 = st.columns(2)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
                ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot2')
                ot4 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot4')
            with colo1:
                ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
                ot3 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot3')
                ot5 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot5')
            outTurn.append(ot0)
            outTurn.append(ot1)
            outTurn.append(ot2)
            outTurn.append(ot3)
            outTurn.append(ot4)
            outTurn.append(ot5)
    elif turns == 6:
        colt0, colt1 = st.columns(2)
        with colt0:
            leng0 = st.number_input('Enter Lengths of Each Run: ', min_value = 1, max_value=2400, key = 'l0')
            leng2 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l2', max_value=2400)
            leng4 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l4', max_value=2400)
            leng6 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l6', max_value=2400)
        with colt1:
            leng1 = st.number_input('', min_value = 1, label_visibility='hidden', key = 'l1', max_value=2400)
            leng3 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l3', max_value=2400)
            leng5 = st.number_input('', min_value = 1, label_visibility='collapsed', key = 'l5', max_value=2400)
        if outlet[0] == True:
            colo0, colo1 = st.columns(2)
            with colo0:
                ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
                ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot2')
                ot4 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot4')
                ot6 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot6')
            with colo1:
                ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
                ot3 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot3')
                ot5 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot5')
            outTurn.append(ot0)
            outTurn.append(ot1)
            outTurn.append(ot2)
            outTurn.append(ot3)
            outTurn.append(ot4)
            outTurn.append(ot5)
            outTurn.append(ot6)
elif pool == True:
    colt0, colt1 = st.columns(2)
    with colt0:
        leng0 = st.number_input('Enter Dimensions of Pool: ', min_value = 1, max_value=2400, key = 'l0')
    with colt1:
        leng1 = st.number_input('', min_value = 1, label_visibility='hidden', max_value=2400, key = 'l01')
    if outlet[0] == True:
        colo0, colo1 = st.columns(2)
        with colo0:
            ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
            ot2 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot2')
        with colo1:
            ot1 = st.number_input('', min_value = 0, max_value = 480, label_visibility='hidden',  key = 'ot1')
            ot3 = st.number_input('', min_value = 0, max_value = 480, label_visibility='collapsed',  key = 'ot3')
        outTurn.append(ot0)
        outTurn.append(ot1)
        outTurn.append(ot2)
        outTurn.append(ot3)

else:
    leng = st.number_input('Enter Length: ', min_value = 1, value = 1, max_value=2400, key = 'l00')
    if outlet[0] == True:
        ot0 = st.number_input('Enter number of outlets:', min_value = 0, max_value = 480, key = 'ot0')
        outTurn.append(ot0)
    straight = True
fin = st.selectbox("Select Finish:", ('SS', 'PS', 'SB', 'ORB', 'BK', 'SPF'), key = 'fin')
col1, col2,col3 = st.columns(3)
with col1:
       st.text('')
       submitted = st.button("Calculate", use_container_width = True)
with col3:
    def clear_text():
        st.session_state['outdoor'] = 'Indoor'
        st.session_state['chann'] = 'PVC Channel'
        st.session_state['ai'] = False
        st.session_state['oltop'] = False
        st.session_state['mod'] = 'S-AG 38'
        if straight == True:
            if st.session_state['l00'] != 1:
                st.session_state['l00'] = 1
        st.session_state['fin'] = 'SS'
    st.text('')
    st.button('Reset', on_click=clear_text, use_container_width = True)


def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
aa = ''
def ssc(mod,leng,fin):
    comp = []
    if outlet[0] == False:
        if outdoor == True or pool == True:
            otlt = math.ceil(float((leng/12)/8))
        else:
            otlt = math.ceil(float((leng/12)/5))
    else:
        otlt = outTurn[0]
        if turn == True or pool == True:
            outTurn.pop(0)
    ccount = 0
    chqt = 0
    grt = ""
    ch = ""
    ec = ""
    ot = ""
    js = ""
    hb = ""
    os = ""
    aj = ""
    key = ""
    stainless = False
    def quantity(a, b):
        for i in range(b):
            comp.append(a)

    def glen(l):
        ar = 0
        am = 0
        rema = []
        l1 = []
        l2 = []
        am0 = 0
        for i in range(len(l)):
            l1 = list(l.values())
            l2 = list(l.keys())
            rema.append(int(leng%l1[i]))
            ar = max(rema)
            am = l1[rema.index(ar)]
            am0 = (l2[rema.index(ar)])
        sagm = int(leng//am) #get count
        sagr = leng%am #remainder
        ae = 0
        ae0 = 0
        for i in range(len(l2)):
            if sagr > ae and sagr < int(l2[i]):
                ae0 = int(int(l2[i]))
                break
            else:
                ae = int(l2[i])
        return([sagm, am0, ae0])

    

    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65") or (mod == "S-AG 100") or (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
        if (mod == "S-AG 38") or (mod == "S-DG 38"):
            if (mod == "S-AG 38"):
                grt = "A 38"
                key = "AKEY"
            else:
                grt = "D 38"
                key = "DKEY"
            ch = "G 38"
            ec = "E 38"
            ot = "S 50"
            js = "GJC 38"
            hb = "HB 32"
            os = "S 32"
            aj = "GAM 38"
        elif (mod == "S-LAG 38") or (mod == "S-LT 38"):
            if (mod == "S-LAG 38"):
                grt = "SA 38"
            else:
                grt = "KA 38"
            ch = "GL 38"
            ec = "EL 38"
            ot = "S 50"
            js = "GJL 38"
            hb = "HB 32"
            os = "SL 32"
            aj = "GAL 38"
            key = "AKEY"
        elif (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65"):
            if (mod == "S-AG 65"):
                grt = "A 65"
                key = "AKEY"
            elif (mod == "S-DG 65"):
                grt = "D 65"
                key = "DKEY"
            else:
                grt = "TA 65"
                key = "AKEY"
            ch = "G 65"
            ec = "E 65"
            ot = "S 50"
            js = "GJC 65"
            hb = "HB 65"
            os = "S 65"
            aj = "GAM 65"
        elif (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65"):
            if (mod == "S-LAG 65"):
                grt = "SA 65"
            elif (mod == "S-LT 65"):
                grt = "KA 65"
            else:
                grt = "RA 65"
            ch = "GL 65"
            ec = "EL 65"
            ot = "S 50"
            js = "GJL 65"
            hb = "HB 65"
            os = "SL 65"
            aj = "GAL 65"
            key = "AKEY"
        elif (mod == "S-AG 100"):
            grt = "A 100"
            ch = "G 100"
            ec = "E 100"
            ot = "S 50"
            js = "GJ 100"
            hb = "HB 65"
            aj = "GA 100"
            key = "AKEY"
        elif (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
            stainless = True
            if (mod == "S-AS 65") or (mod == "S-AS 99"):
                grt = "SA 65"
            else:
                grt = "RA 65"
            ch = "SC 65"
            if (mod == "S-AS 65") or (mod == "S-LTIFAS 65"):
                os = "LF 65"
                ot = "TNAS"
                hb = "HB 65"
            else:
                os = "LF 99"
            aj = "SLA 65"
            js = "GJS 65"
            ec = "SE 65"
            key = "AKEY"
        else:
            stainless = True
            grt = "TA 65"
            ch = "TC 65"
            if (mod == "S-TIFAS 65"):
                os = "HF 65"
                ot = "TNAS"
                hb = "HB 65"
            else:
                os = "HF 99"
            aj = "SHA 65"
            js = "TJS 65"
            ec = "TE 65"
            key = "AKEY"

        #0"-96"
        if ((leng > 0) and (leng < 97)):
            if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                if (leng < 37):
                    quantity(grt + "36 " + str(fin), 1)
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38"):
                        quantity(ch + "36", 1)
                    for i in range(0,2):
                        comp2.append("CT 36 " + str(fin)) 
                elif (leng > 36) and (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38"):
                        quantity(ch + "48", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                elif (leng > 48) and (leng < 61):
                    quantity(grt + "60 " + str(fin), 1)
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38"):
                        quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                elif (leng > 60) and (leng < 73):
                    quantity(grt + "72 " + str(fin), 1)
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38"):
                        quantity(ch + "72", 1)
                    for i in range(0,2):
                        comp2.append("CT 72 " + str(fin))
                else:
                    quantity(grt + "48 " + str(fin), 2)
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38"):
                        quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 96 " + str(fin))
            if (mod == "S-DG 65") or (mod == "S-DG 38"):
                if (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    quantity(ch + "48", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                elif (leng > 48) and (leng < 61):
                    quantity(grt + "60 " + str(fin), 1)
                    quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                else:
                    quantity(grt + "96 " + str(fin), 1)
                    quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 96 " + str(fin))
            if (mod == "S-LT 65") or (mod == "S-LT 38"):               
                if (leng < 37):
                    quantity(grt + "36 " + str(fin), 1)
                    quantity(ch + "36", 1)
                    for i in range(0,2):
                        comp2.append("CT 36 " + str(fin))
                elif (leng > 36) and (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    quantity(ch + "48", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                elif (leng > 48) and (leng < 61):
                    quantity(grt + "60 " + str(fin), 1)
                    quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                elif (leng > 60) and (leng < 73):
                    quantity(grt + "72 " + str(fin), 1)
                    quantity(ch + "72", 1)
                    for i in range(0,2):
                        comp2.append("CT 72 " + str(fin))
                else:
                    quantity(grt + "96 " + str(fin), 1)
                    quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 96 " + str(fin))
            if (mod == "S-LTIF 65") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
                if (leng < 37):
                    quantity(grt + "36 " + str(fin), 1)
                    if (mod == "S-LTIF 65"):
                        quantity(ch + "36", 1)
                    for i in range(0,2):
                        comp2.append("CT 36 " + str(fin))
                elif (leng > 36) and (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    if (mod == "S-LTIF 65"):
                        quantity(ch + "48", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                elif (leng > 48) and (leng < 61):
                    quantity(grt + "60 " + str(fin), 1)
                    if (mod == "S-LTIF 65"):
                        quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                elif (leng > 60) and (leng < 73):
                    quantity(grt + "36 " + str(fin), 2)
                    if (mod == "S-LTIF 65"):    
                        quantity(ch + "72", 1)
                    for i in range(0,2):
                        comp2.append("CT 72 " + str(fin))
                else:
                    quantity(grt + "48 " + str(fin), 2)
                    if (mod == "S-LTIF 65"):
                        quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 96 " + str(fin))
            if (mod == "S-TIF 65") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
                if (leng < 41):
                    quantity(grt + "40 " + str(fin), 1)
                    if (mod == "S-TIF 65"):
                        quantity(ch + "40", 1)
                    for i in range(0,2):
                        comp2.append("CT 40 " + str(fin))
                elif (leng > 40) and (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    if (mod == "S-TIF 65"):
                        quantity(ch + "48", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                elif (leng > 48) and (leng < 61):
                    quantity(grt + "60 " + str(fin), 1)
                    if (mod == "S-TIF 65"):
                        quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                elif (leng > 60) and (leng < 81):
                    quantity(grt + "40 " + str(fin), 2)
                    if (mod == "S-TIF 65"):
                        quantity(ch + "80", 1)
                    for i in range(0,2):
                        comp2.append("CT 80 " + str(fin))
                else:
                    quantity(grt + "48 " + str(fin), 2)
                    if (mod == "S-TIF 65"):
                        quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 96 " + str(fin))
            if (mod == "S-AG 100"):
                if (leng < 49):
                    quantity(grt + "48 " + str(fin), 1)
                    quantity(ch + "48", 1)
                else:
                    quantity(grt + "48 " + str(fin), 2)
                    quantity(ch + "96", 1)

        #97"-192"
        if ((leng > 96) and (leng < 193)):
            leng1 = leng/2
            if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65") or (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
                #108" MAX
                if(leng1 in arange(48.0,54.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 1)
                    elif (mod == "S-DG 65") or (mod == "S-DG 38") or (mod == "S-LT 38") or (mod == "S-LT 65"):
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 1)
                    else:
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 1)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "48", 1)
                        quantity(ch + "60", 1)
                    for i in range(0,2):
                        comp2.append("CT 48 " + str(fin))
                        comp2.append("CT 60 " + str(fin))
                    
                #120" MAX
                if(leng1 in arange(54.5,60.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "60 "  + str(fin), 2)
                    elif (mod == "S-DG 65") or (mod == "S-DG 38") or (mod == "S-LT 38") or (mod == "S-LT 65"):
                        quantity(grt + "60 " + str(fin), 2)
                    else:
                        quantity(grt + "60 " + str(fin), 2)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "60", 2)
                    for i in range(0,4):
                        comp2.append("CT 60 " + str(fin))
                #132" MAX
                if(leng1 in arange(60.5,66.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-LT 65") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "60 " + str(fin), 1)
                        quantity(grt + "72 " + str(fin), 1)
                    elif (mod == "S-DG 65") or (mod == "S-DG 38") or (mod == "S-LTIF 65") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
                        quantity(grt + "48 " + str(fin), 3)
                    else:
                        quantity(grt + "40 " + str(fin), 1)
                        quantity(grt + "48 " + str(fin), 2)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "60", 1)
                        quantity(ch + "72", 1)
                    for i in range(0,2):
                        comp2.append("CT 60 " + str(fin))
                        comp2.append("CT 72 " + str(fin))
                #144" MAX    
                if(leng1 in arange(66.5,72.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-LT 65") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "72 " + str(fin), 2)
                        
                    elif (mod == "S-DG 65") or (mod == "S-DG 38") or (mod == "S-LTIF 65") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
                        quantity(grt + "48 " + str(fin), 3)
                    else:
                        quantity(grt + "48 " + str(fin), 3)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "72", 2)
                    for i in range(0,4):
                        comp2.append("CT 72 " + str(fin))
                #168" MAX
                if(leng1 in arange(72.5,84.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-LT 65") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 2)
                    elif (mod == "S-DG 65") or (mod == "S-DG 38"):
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 2)
                    else:
                        quantity(grt + "48 " + str(fin), 1)
                        quantity(grt + "60 " + str(fin), 2)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "72", 1)
                        quantity(ch + "96", 1)
                    for i in range(0,2):
                        comp2.append("CT 72 " + str(fin))
                        comp2.append("CT 96 " + str(fin))
                #192" MAX
                if(leng1 in arange(84.5,96.5,0.5)):
                    if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                        quantity(grt + "60 " + str(fin), 2)
                        quantity(grt + "72 " + str(fin), 1)
                    elif (mod == "S-DG 65") or (mod == "S-DG 38") or (mod == "S-LT 38") or (mod == "S-LT 65"):
                        quantity(grt + "96 "  + str(fin), 2)
                    else:
                        quantity(grt + "48 " + str(fin), 4)
                    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                        quantity(ch + "96", 2)
                    for i in range(0,4):
                        comp2.append("CT 96 " + str(fin))
                ccount = 2
            #144" MAX
            if (mod == "S-AG 100"):
                if (leng1 in arange(48.0,72.5,0.5)):
                    quantity(grt + "48 " + str(fin), 3)
                    quantity(ch + "48", 3)
                    ccount = 3
                elif (leng1 in arange(72.5,96.5,0.5)):
                    quantity(grt + "48 " + str(fin), 4)
                    quantity(ch + "96", 2)
                    ccount = 2

        if leng > 192:
            #S-AG
            if (mod == "S-AG 65") or (mod == "S-AG 38") or (mod == "S-LAG 65") or (mod == "S-LAG 38") or (mod == "S-AS 65") or (mod == "S-AS 99"):
                l1 = {"36": 35.875, "48": 47.875, "60": 59.875, "72": 71.875}
                samplist = glen(l1)
                sl0 = 0
                sl1 = ""
                if samplist[1] == "36":
                    sl1 = "72"
                    if samplist[0]%2 == 0:
                        sl0 = int(samplist[0]/2)
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        sl0 = int(samplist[0]-1)/2
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[1]) +" "+ str(fin), 1)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                else:    
                    quantity(grt + str(samplist[1]) +" "+ str(fin), samplist[0])
                    quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                    if samplist[1] == '48':
                        sl1 = '96'
                        if samplist[0]%2 ==0:
                            sl0 = int(samplist[0]/2)
                            for i in range(0,int(sl0 * 2)):
                                comp2.append("CT "+ str(sl1) + " " + str(fin))
                            for i in range(0,2):
                                comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                        else:
                            sl0 = int(samplist[0]-1)/2
                            for i in range(0,int(sl0 * 2)):
                                comp2.append("CT "+ str(sl1) + " " + str(fin))
                            for i in range(0,2):
                                comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                                comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        for i in range(0,int(samplist[0] * 2)):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
            #S-LT
            if (mod == "S-LT 38") or (mod == "S-LT 65"):
                l1 = {"36": 35.875, "48": 47.875, "60": 59.875, "72": 71.875, "96": 95.875}
                samplist = glen(l1)
                sl0 = 0
                sl1 = ""
                if samplist[1] == "36" or samplist[1] == "48":
                    if samplist[1] == "36":
                        sl1 = "72"
                    else:
                        sl1 = "96"
                    if samplist[0]%2 == 0:
                        sl0 = int(samplist[0]/2)
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        sl0 = int(samplist[0]-1)/2
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[1]) +" "+ str(fin), 1)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                        
                else:    
                    quantity(grt + str(samplist[1]) +" "+ str(fin), samplist[0])
                    quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                    for i in range(0,int(samplist[0] * 2)):
                        comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                    for i in range(0,2):
                        comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
            #S-DG
            if (mod == "S-DG 65") or (mod == "S-DG 38"):
                l1 = {"48": 47.875, "60": 59.875, "96": 95.875}
                samplist = glen(l1)
                sl0 = 0
                sl1 = ""
                if samplist[1] == "48":
                    sl1 = "96"
                    if samplist[0]%2 == 0:
                        sl0 = int(samplist[0]/2)
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        sl0 = int(samplist[0]-1)/2
                        quantity(grt + str(sl1) +" "+ str(fin), sl0)
                        quantity(grt + str(samplist[1]) +" "+ str(fin), 1)
                        quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                else:    
                    quantity(grt + str(samplist[1]) +" "+ str(fin), samplist[0])
                    quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                    for i in range(0,int(samplist[0] * 2)):
                        comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                    for i in range(0,2):
                        comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
            #S-LTIF 65
            if (mod == "S-LTIF 65") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
                l1 = {"36": 35.875, "48": 47.875, "60": 59.875}
                samplist = glen(l1)
                quantity(grt + str(samplist[1]) +" "+ str(fin), samplist[0])
                quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                if samplist[1] == '36' or samplist[1] == '48':
                    if samplist[1] == '36':
                        sl1 = '72'
                    else:
                        sl1 = '96'
                    if samplist[0]%2 ==0:
                        sl0 = int(samplist[0]/2)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        sl0 = int(samplist[0]-1)/2
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                else:
                    for i in range(0,int(samplist[0] * 2)):
                        comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                    for i in range(0,2):
                        comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
            #S-TIF 65
            if (mod == "S-TIF 65") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
                l1 = {"40": 39.875, "48": 47.875, "60": 59.875}
                samplist = glen(l1)
                quantity(grt + str(samplist[1]) +" "+ str(fin), samplist[0])
                quantity(grt + str(samplist[2]) +" "+ str(fin), 1)
                if samplist[1] == '40' or samplist[1] == '48':
                    if samplist[1] == '40':
                        sl1 = '80'
                    else:
                        sl1 = '96'
                    if samplist[0]%2 ==0:
                        sl0 = int(samplist[0]/2)
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                    else:
                        sl0 = int(samplist[0]-1)/2
                        for i in range(0,int(sl0 * 2)):
                            comp2.append("CT "+ str(sl1) + " " + str(fin))
                        for i in range(0,2):
                            comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                            comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
                else:
                    for i in range(0,int(samplist[0] * 2)):
                        comp2.append("CT "+ str(samplist[1]) + " " + str(fin))
                    for i in range(0,2):
                        comp2.append("CT "+ str(samplist[2]) + " " + str(fin))
            #S-AG 100
            if (mod == "S-AG 100"):
                g = math.ceil(float(leng/47.25))
                quantity(grt + "48 " + str(fin), g)
                g1 = math.ceil(float(leng/47.5))
                if g1%2 == 0:
                    quantity(ch + "96", int(g1/2))
                    ccount = (int(g1/2))
                else:
                    quantity(ch + "96", int((g1-1)/2))
                    quantity(ch + "48", 1)
                    ccount = g1

            if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65"):
                l1 = {"36": 33.8125, "48": 45.8125, "60": 57.8125, "72": 69.8125, "96": 93.8125}
                samplist = glen(l1)
                sl1 = ""
                sl0 = 0
                if samplist[1] == "36" or samplist[1] == "48":
                    if samplist[1] == "36":
                        sl1 = "72"
                    elif samplist[1] == "48":
                        sl1 = "96"
                    if samplist[0]%2 == 0:
                        sl0 = int((samplist[0])/2)
                        quantity(ch + str(sl1), sl0)
                        quantity(ch + str(samplist[2]), 1)
                        ccount = sl0+1

                    else:
                        sl0 = int((samplist[0])-1)/2
                        quantity(ch + str(sl1), sl0)
                        quantity(ch + str(samplist[1]), 1)
                        quantity(ch + str(samplist[2]), 1)
                        ccount = sl0 + 2

                else:
                        quantity(ch + str(samplist[1]), samplist[0])
                        quantity(ch + str(samplist[2]), 1)
                        ccount = samplist[0] + 1
            
        if (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
                asch = []
                cch = ""
                l1 = {"36": 36, "48": 48, "60": 60, "72": 72, "96": 96}
                if pool == False and turns == 0:
                    if (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
                        asch = [28, 40, 52, 64, 88]
                        cch = "LC 65"
                    else:
                        asch = [32, 40, 52, 72, 88]
                        cch = "HC 65"
                    asch2 = []
                    asch3 = []
                    bn = False
                    for i in range(len(asch)):
                        if leng < (asch[i] + 8) or leng == (asch[i] + 8):
                            bn = True
                            quantity(cch + (str(asch[i]))+ " " + str(fin), 1)
                            break
                    if bn == False:
                        for i in range(len(asch)):
                            asch1 = leng - asch[i]
                            asch2.append(math.ceil(asch1/48))
                            asch3.append((asch1%48))
                        asch4 = asch3.index(max(asch3))
                        quantity(cch + (str(asch[asch4]))+ " " + str(fin), 1)
                        if (((asch2[asch4]) % 2) == 0):
                            quantity(ch + "96 " + str(fin), int(asch2[asch4]/2))
                            chqt = int(asch2[asch4]/2)
                        else:
                            quantity(ch + "96 " + str(fin), int((asch2[asch4]-1)/2))
                            quantity(ch + "48 " + str(fin), 1)
                            chqt = int(((asch2[asch4]-1)/2)+1)
                    
                else:
                    chqt1 = (math.ceil((leng/48)))
                    if (chqt1 % 2 == 0):
                        quantity(ch + "96 " + str(fin), int(chqt1/2))
                        chqt = int(chqt1/2)
                    else:
                        quantity(ch + "96 " + str(fin), int((chqt1-1)/2))
                        quantity(ch + "48 " + str(fin), 1)
                        chqt = int(((chqt1-1)/2)+1)


    if ((otlt//2) < 1):
        kq = 1
    else:
        kq = (otlt//2)
    if outlet[0] == False:
        if (otlt < 1):
            otlt = 1
    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65") or (mod == "S-AG 100"):
        if outdoor == True:
            if turns == 0 and pool == False:
                quantity(ec, 2)
                quantity("F 65", otlt)
                quantity(js, ccount-1)
                quantity(hb, otlt)
                if (mod != "S-AG 100"):
                    quantity(os, otlt)
                quantity(key, kq)
            else:
                quantity("F 65", otlt)
                quantity(js, ccount-1)
                quantity(hb, otlt)
                if (mod != "S-AG 100"):
                    quantity(os, otlt)
                quantity(key, kq)
        else:
            if turns == 0 and pool == False:
                quantity(ec, 2)
                quantity(ot, otlt)
                quantity(js, ccount-1)
                quantity(hb, otlt)
                if (mod != "S-AG 100"):
                    quantity(os, otlt)
                quantity(key, kq)
                for i in range(0,otlt):
                    comp1.append("Drain Body")
            else:
                quantity(ot, otlt)
                quantity(js, ccount-1)
                quantity(hb, otlt)
                if (mod != "S-AG 100"):
                    quantity(os, otlt)  
                quantity(key, kq)
                for i in range(0,otlt):
                    comp1.append("Drain Body")
    else:
        if (mod == "S-AS 65") or (mod == "S-LTIFAS 65") or (mod == "S-TIFAS 65"):
            quantity(os + " " + str(fin), otlt)
            quantity(js, (otlt*2)+(chqt+1))
            quantity("ZSIKA", math.ceil((int((otlt*2)+(chqt+1)))/6))
            quantity(ot, otlt)
            quantity(hb, otlt)
            quantity(key, kq)
        else:
            quantity(os + " " + str(fin), otlt)
            quantity(js, (otlt*2)+(chqt+1))
            quantity("ZSIKA", math.ceil((int((otlt*2)+(chqt+1)))/6))
            quantity(key, kq)
        for i in range(0,otlt):
            comp1.append("Drain Body")
    a = dict(Counter(comp))
    aa = list(a.keys())
    aa1.append(aa)
    ab = list(a.values())
    ac = {'Components': aa, 'Quantities': ab}
    final = (pd.DataFrame.from_dict(ac))
    return(final)

def ajs(mod, turn):
    comp = []
    if (mod == "S-AG 38") or (mod == "S-DG 38") or (mod == "S-LAG 38") or (mod == "S-LT 38") or (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65") or (mod == "S-LAG 65") or (mod == "S-LT 65") or (mod == "S-LTIF 65") or (mod == "S-DG 65") or (mod == "S-AG 100") or (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99") or (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
        if (mod == "S-AG 38") or (mod == "S-DG 38"):
            ec = "E 38"
            aj = "GAM 38"
        elif (mod == "S-LAG 38") or (mod == "S-LT 38"):
            ec = "EL 38"
            aj = "GAL 38"
        elif (mod == "S-AG 65") or (mod == "S-DG 65") or (mod == "S-TIF 65"):
            ec = "E 65"
            aj = "GAM 65"
        elif (mod == "S-AG 100"):
            ec = "E 100"
            aj = "GA 100"
        elif (mod == "S-AS 65") or (mod == "S-AS 99") or (mod == "S-LTIFAS 65") or (mod == "S-LTIFAS 99"):
            ec = "SE 65 " + str(fin)
            aj = "SLA 65 " + str(fin)
        elif (mod == "S-TIFAS 65") or (mod == "S-TIFAS 99"):
            ec = "TE 65 " + str(fin)
            aj = "SHA 65 " + str(fin)
        else:
            ec = "EL 65"
            aj = "GAL 65"
    def quantity(a, b):
        for i in range(b):
            comp.append(a)
    if turns > 0:
        quantity(ec, 2)
        quantity(aj, turn)
        if outdoor == False:    
            for i in range(len(comp1)):
                comp.append(comp1[i])
            if (mod != "S-AG 100"):
                for i in range(len(comp2)):
                    comp.append(comp2[i])
        else:
            if stainless == True:
                for i in range(len(comp1)):
                    comp.append(comp1[i])
        comp2.clear()
        comp1.clear()
    elif pool == True:
        quantity(aj, 4)
        if outdoor == False:
            for j in range (0,2):
                for i in range(len(comp1)):
                    comp.append(comp1[i])
            comp1.clear()
        else:
            if stainless == True:
                for i in range(0,2):
                    for i in range(len(comp1)):
                        comp.append(comp1[i])
        comp2.clear()

    else:
        if outdoor == False:    
            for i in range(len(comp1)):
                comp.append(comp1[i])
            if (mod != "S-AG 100"):
                for i in range(len(comp2)):
                    comp.append(comp2[i])
        else:
            if stainless == True:
                for i in range(len(comp1)):
                    comp.append(comp1[i])
        
        comp2.clear()
        comp1.clear()
    a = dict(Counter(comp))
    aa = list(a.keys())
    ab = list(a.values())
    ac = {'Components': aa, 'Quantities': ab}
    final = (pd.DataFrame.from_dict(ac))
    return(final)
    
sah = {'A': ' - Wedge Wire Grate', 
       'SA': ' - Wedge Wire Grate',
       'D': ' - Circle Pattern Grate',
       'KA': ' - Offset Slotted Pattern Grate',
       'TA': ' - Tile Insert Frame',
       'RA': ' - Tile Insert Frame', 
       'G': ' - PVC Channel', 
       'GL': ' - PVC Channel',
       'E': ' - PVC Stop End',
       'EL': ' - PVC Stop End', 
       'S 50': ' - Threaded Outlet', 
       'GJC': ' - Joiner Strip',
       'GJL': ' - Joiner Strip',
       'GJ': ' - Joiner Strip', 
       'HB': ' - Hair Basket', 
       'S': ' - Outlet Section',
       'SL': ' - Outlet Section', 
       'F': ' - No-Hub Outlet', 
       'GAM': ' - 90° Angle Joiner',
       'GAL': ' - 90° Angle Joiner',
       'GA': ' - 90° Angle Joiner', 
       'CT': ' - Channel Trim', 
       "AKEY": ' - Lift Out Key', 
       'DKEY': 'Lift Out Key',
       'LC': ' - Closed Ended Stainless Steel Channel',
       'HC': ' - Closed Ended Stainless Steel Channel',
       'SC': ' - Open Ended Stainless Steel Channel',
       'TC': ' - Open Ended Stainless Steel Channel',
       'LF': ' - Outlet Section',
       'HF': ' - Outlet Section',
       'SLA': ' - 90° Angle Joiner',
       'SHA': ' - 90° Angle Joiner',
       'ZSIKA': ' - M-1 All Purpose Sealant',
       'GJS': ' - Joiner Strip',
       'TJS': ' - Joiner Strip',
       'TNAS': ' - Threaded Outlet',
       'SE': ' - Stainless Steel Stop End',
       'TE': ' - Stainless Steel Stop End',
       'Drain': ' - Clamp Down'}

if submitted:
    def ta(leng,f, z):
        if z == 0:
            st.markdown('**Additional Components**')
        else:
            st.markdown('**' + (str(mod) + str(leng) + ' ' + str(fin))+ '**')
        st.write((f.to_markdown(index=False))) 
    def li(leng,f, z):
        if z == 0:
            st.markdown('**Additional Components**')
        else:
            st.markdown('**' + (str(mod) + str(leng) + ' ' + str(fin))+ '**')
        a1 = (f.values.tolist())
        for i in range(0, len(a1)):
            cp = ((str(a1[i][0])))
            qt = ((str(a1[i][1])))
            fl = ((str(a1[i][0])).split(' ')[0])
            if fl == 'S':
                if ((str(a1[i][0])).split(' ')[1]) == '50':
                    fl = 'S 50'
                else:
                    fl = 'S'
            for j,k in sah.items():
                if fl == j:
                    st.markdown("(" + qt + ") " + '**'+ cp + '**'+ k)
    tab1, tab2 = st.tabs(['Table', 'List'])
    #Straight run
    if turn == True and turns != 1:
        if turns == 2:
            f0 = ssc(mod,leng0,fin)
            f1 = ssc(mod,leng1,fin)
            f2 = ssc(mod,leng2,fin)
            aj0 = ajs(mod,turns)
            if mod == 'S-AG 100':
                if leng0 == 3 and leng1 == 7 and leng2 == 22:
                    st.snow()
                    st.balloons()
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0,1)
                with colfin2:
                    ta(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0,1)   
                with colfin4:
                    li(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                "---"
                # st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng2,f2,1)
                with colfin2:
                    ta(leng0,aj0,0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng2,f2,1)   
                with colfin4:
                    li(leng3,aj0,0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
        if turns == 3:
            f0 = ssc(mod,leng0,fin)
            f1 = ssc(mod,leng1,fin)
            f2 = ssc(mod,leng2,fin)
            f3 = ssc(mod,leng3,fin)
            aj0 = ajs(mod,turns)
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0,1)
                with colfin2:
                    ta(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0,1)   
                with colfin4:
                    li(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng2,f2,1)
                with colfin2:
                    ta(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng2,f2,1)   
                with colfin4:
                    li(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------')
        if turns == 4:
            f0 = ssc(mod,leng0,fin)
            f1 = ssc(mod,leng1,fin)
            f2 = ssc(mod,leng2,fin)
            f3 = ssc(mod,leng3,fin)
            f4 = ssc(mod,leng4,fin)
            aj0 = ajs(mod,turns)
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0,1)
                with colfin2:
                    ta(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0,1)   
                with colfin4:
                    li(leng1,f1,1)
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng2,f2,1)
                with colfin2:
                    ta(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng2,f0,1)   
                with colfin4:
                    li(leng3,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng4,f4,1)
                with colfin2:
                    ta(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng4,f0,1)   
                with colfin4:
                    li(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')

        if turns == 5:
            f0 = ssc(mod,leng0,fin)
            f1 = ssc(mod,leng1,fin)
            f2 = ssc(mod,leng2,fin)
            f3 = ssc(mod,leng3,fin)
            f4 = ssc(mod,leng4,fin)
            f5 = ssc(mod,leng5,fin)
            aj0 = ajs(mod,turns)
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0,1)
                with colfin2:
                    ta(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0,1)   
                with colfin4:
                    li(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng2,f2,1)
                with colfin2:
                    ta(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng2,f0,1)   
                with colfin4:
                    li(leng3,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng4,f4,1)
                with colfin2:
                    ta(leng5,f5,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng4,f0,1)
                with colfin4:
                    li(leng5,f5,1)      
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------')
            

        if turns == 6:
            f0 = ssc(mod,leng0,fin)
            f1 = ssc(mod,leng1,fin)
            f2 = ssc(mod,leng2,fin)
            f3 = ssc(mod,leng3,fin)
            f4 = ssc(mod,leng4,fin)
            f5 = ssc(mod,leng5,fin)
            f6 = ssc(mod,leng6,fin)
            aj0 = ajs(mod, turns)
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0,1)
                with colfin2:
                    ta(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0,1)   
                with colfin4:
                    li(leng1,f1,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng2,f2,1)
                with colfin2:
                    ta(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng2,f2,1)   
                with colfin4:
                    li(leng3,f3,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng4,f4,1)
                with colfin2:
                    ta(leng5,f5,1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng4,f4,1)
                with colfin4:
                    li(leng5,f5, 1)      
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng6,f6, 1)
                with colfin2:
                    ta(leng0, aj0, 0)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng6,f6, 1) 
                with colfin4:
                    li(leng0, aj0, 0)  
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
    elif pool == True or turns == 1:
        f0 = ssc(mod,leng0,fin)
        f1 = ssc(mod,leng1,fin)
        aj0 = ajs(mod,turns)
        b = 1
        if pool == True:
            b = 2
        for i in range(0,b):
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng0,f0, 1)
                with colfin2:
                    ta(leng1,f1, 1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng0,f0, 1)
                with colfin4:
                    li(leng1,f1, 1)
                st.text('\n')
                st.text('\n')
                st.text('--------------------------                 --------------------------')
        with tab1:
            colfin1, colfin2 = st.columns(2)
            with colfin1:
                ta(leng0, aj0, 0)
            st.text('\n')
            st.text('\n')
            st.text('--------------------------')
        with tab2:
            colfin3, colfin4 = st.columns(2)
            with colfin3:
                li(leng0, aj0, 0)
            st.text('\n')
            st.text('\n')
            st.text('--------------------------')
    else:
        f0 = ssc(mod,leng,fin)
        aj0 = ajs(mod,turns)
        if outdoor == False:
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng,f0,1)
                with colfin2:
                    ta(leng,aj0,0)
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:    
                    li(leng,f0,1)
                with colfin4:
                    li(leng,aj0,0)
        else:
            with tab1:
                colfin1, colfin2 = st.columns(2)
                with colfin1:
                    ta(leng,f0,1)
                if stainless == True:
                    with colfin2:
                        ta(leng,aj0,0)
            with tab2:
                colfin3, colfin4 = st.columns(2)
                with colfin3:
                    li(leng,f0,1)
                if stainless == True:
                    with colfin4:
                        li(leng,aj0, 0)
