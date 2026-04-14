
 
//Mac

clear all 
 
cd /Users/pjindapon/Dropbox/Stata/publicgood

import excel "/Users/pjindapon/Dropbox/Stata/publicgood/contributions.xlsx", sheet("contributions") firstrow



//PC:

clear all 

cd "C:\Users\jinda\Dropbox\Stata\publicgood"

import excel "/Users/jinda/Dropbox/Stata/publicgood/contributions.xlsx", sheet("contributions") firstrow




/////////////////////////




gen othercont=(payoff-25+0.6*contribution)/0.4
gen othercontaverage=(payoff-25+0.6*contribution)/1.3

gen morethanaverage=(contribution>othercontaverage)
gen lessthanaverage=(contribution<othercontaverage)


gen diffcont=contribution-othercontaverage
gen contmore=diffcont*morethanaverage
gen contless=-diffcont*lessthanaverage


gen sessionnumber=1
replace sessionnumber=2 if session_code=="umbzdj98"
replace sessionnumber=3 if session_code=="sylq2syi"
replace sessionnumber=4 if session_code=="iiu3xixz"
replace sessionnumber=5 if session_code=="j3ki5tli"
replace sessionnumber=6 if session_code=="r5dj4yfl"
replace sessionnumber=7 if session_code=="irrzlgk2"
replace sessionnumber=8 if session_code=="sa7mprty"
replace sessionnumber=9 if session_code=="6ucza025"
replace sessionnumber=10 if session_code=="6sdkxl2q"

gen subject_id=sessionnumber*100+participant_id 


gen segmentnumber=1
replace segmentnumber=2 if segment=="supergame2"
replace segmentnumber=3 if segment=="supergame3"
replace segmentnumber=4 if segment=="supergame4"
replace segmentnumber=5 if segment=="supergame5"


gen segment1=(segmentnumber==1) 
gen segment2=(segmentnumber==2) 
gen segment3=(segmentnumber==3) 
gen segment4=(segmentnumber==4) 
gen segment5=(segmentnumber==5) 


gen round1=(round==1) 
gen round2=(round==2) 
gen round3=(round==3) 
gen round4=(round==4) 
gen round5=(round==5) 
gen round6=(round==6) 
gen round7=(round==7) 

gen segment1round1=segment1*round1


gen period=round
replace period=round+3 if segmentnumber==2
replace period=round+7 if segmentnumber==3
replace period=round+10 if segmentnumber==4
replace period=round+17 if segmentnumber==5

gen period2=period^2

sort subject_id period

xtset subject_id period



gen cont_L1=contribution[_n-1]
replace cont_L1=. if period==1
gen dcont=contribution-cont_L1



gen othercont_L1=othercont[_n-1]
replace othercont_L1=. if period==1

gen more_L1=morethanaverage[_n-1]
replace more_L1=. if period==1

gen less_L1=lessthanaverage[_n-1]
replace less_L1=. if period==1

gen contmore_L1=contmore[_n-1]
replace contmore_L1=. if period==1

gen contless_L1=contless[_n-1]
replace contless_L1=. if period==1


xtabond contribution  contmore_L1 contless_L1 round1  round2 round3 round4 round5 segmentnumber if treatment==1  , lags(2) twostep maxldep(4) maxlags(4) 

estat sargan


xtabond contribution contmore_L1 contless_L1  round1  round2 round3 round4 round5  segmentnumber  if treatment==1 , lags(2) twostep maxldep(4) maxlags(4)    vce(robust)

estat abond


outreg2 using "Table DP1", tex dec(3) stats(coef se) symbol(***,**,*) nor2 noni label ctitle(Treatment 1) nonotes replace

test contmore_L1 + contless_L1 =0

test round1 + round2 =0



xtabond contribution contmore_L1 contless_L1  round1  round2 round3 round4 round5 segmentnumber  if treatment==2 , lags(2) twostep maxldep(4) maxlags(4) 

estat sargan


xtabond contribution contmore_L1 contless_L1  round1  round2 round3 round4 round5 segmentnumber  if treatment==2 , lags(2) twostep maxldep(4) maxlags(4)    vce(robust)

outreg2 using "Table DP1", tex dec(3) stats(coef se) symbol(***,**,*) nor2 noni label ctitle(Treatment 2) nonotes

estat abond

test contmore_L1 + contless_L1 =0

test round1 + round2 =0






