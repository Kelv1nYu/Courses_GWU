#delimit ;

   infix
      year     1 - 20
      spevwork 21 - 40
      spwrkslf 41 - 60
      childs   61 - 80
      age      81 - 100
      educ     101 - 120
      sex      121 - 140
      income   141 - 160
      rincome  161 - 180
      sphrs2   181 - 200
      sphrs1   201 - 220
      id_      221 - 240
      wrkstat  241 - 260
      hrs1     261 - 280
      hrs2     281 - 300
      evwork   301 - 320
      wrkslf   321 - 340
      divorce  341 - 360
      spwrksta 361 - 380
      ballot   381 - 400
using GSS.dat;

label variable year     "Gss year for this respondent                       ";
label variable spevwork "Spouse ever work as long as a year";
label variable spwrkslf "Spouse self-emp. or works for somebody";
label variable childs   "Number of children";
label variable age      "Age of respondent";
label variable educ     "Highest year of school completed";
label variable sex      "Respondents sex";
label variable income   "Total family income";
label variable rincome  "Respondents income";
label variable sphrs2   "No. of hrs spouse usually works a week";
label variable sphrs1   "Number of hrs spouse worked last week";
label variable id_      "Respondent id number";
label variable wrkstat  "Labor force status";
label variable hrs1     "Number of hours worked last week";
label variable hrs2     "Number of hours usually work a week";
label variable evwork   "Ever work as long as one year";
label variable wrkslf   "R self-emp or works for somebody";
label variable divorce  "Ever been divorced or separated";
label variable spwrksta "Spouse labor force status";
label variable ballot   "Ballot used for interview";


label define gsp001x
   9        "No answer"
   8        "Don't know"
   2        "No"
   1        "Yes"
   0        "Not applicable"
;
label define gsp002x
   9        "No answer"
   8        "Don't know"
   2        "Someone else"
   1        "Self-employed"
   0        "Not applicable"
;
label define gsp003x
   9        "Dk na"
   8        "Eight or more"
;
label define gsp004x
   99       "No answer"
   98       "Don't know"
   89       "89 or older"
;
label define gsp005x
   99       "No answer"
   98       "Don't know"
   97       "Not applicable"
;
label define gsp006x
   2        "Female"
   1        "Male"
;
label define gsp007x
   99       "No answer"
   98       "Don't know"
   13       "Refused"
   12       "$25000 or more"
   11       "$20000 - 24999"
   10       "$15000 - 19999"
   9        "$10000 - 14999"
   8        "$8000 to 9999"
   7        "$7000 to 7999"
   6        "$6000 to 6999"
   5        "$5000 to 5999"
   4        "$4000 to 4999"
   3        "$3000 to 3999"
   2        "$1000 to 2999"
   1        "Lt $1000"
   0        "Not applicable"
;
label define gsp008x
   99       "No answer"
   98       "Don't know"
   13       "Refused"
   12       "$25000 or more"
   11       "$20000 - 24999"
   10       "$15000 - 19999"
   9        "$10000 - 14999"
   8        "$8000 to 9999"
   7        "$7000 to 7999"
   6        "$6000 to 6999"
   5        "$5000 to 5999"
   4        "$4000 to 4999"
   3        "$3000 to 3999"
   2        "$1000 to 2999"
   1        "Lt $1000"
   0        "Not applicable"
;
label define gsp009x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp010x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp011x
   9        "No answer"
   8        "Other"
   7        "Keeping house"
   6        "School"
   5        "Retired"
   4        "Unempl, laid off"
   3        "Temp not working"
   2        "Working parttime"
   1        "Working fulltime"
   0        "Not applicable"
;
label define gsp012x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp013x
   99       "No answer"
   98       "Don't know"
   -1       "Not applicable"
;
label define gsp014x
   9        "No answer"
   8        "Don't know"
   2        "No"
   1        "Yes"
   0        "Not applicable"
;
label define gsp015x
   9        "No answer"
   8        "Don't know"
   2        "Someone else"
   1        "Self-employed"
   0        "Not applicable"
;
label define gsp016x
   9        "No answer"
   8        "Don't know"
   2        "No"
   1        "Yes"
   0        "Not applicable"
;
label define gsp017x
   9        "No answer"
   8        "Other"
   7        "Keeping house"
   6        "School"
   5        "Retired"
   4        "Unempl, laid off"
   3        "Temp not working"
   2        "Working parttime"
   1        "Working fulltime"
   0        "Not applicable"
;
label define gsp018x
   4        "Ballot d"
   3        "Ballot c"
   2        "Ballot b"
   1        "Ballot a"
   0        "Not applicable"
;


label values spevwork gsp001x;
label values spwrkslf gsp002x;
label values childs   gsp003x;
label values age      gsp004x;
label values educ     gsp005x;
label values sex      gsp006x;
label values income   gsp007x;
label values rincome  gsp008x;
label values sphrs2   gsp009x;
label values sphrs1   gsp010x;
label values wrkstat  gsp011x;
label values hrs1     gsp012x;
label values hrs2     gsp013x;
label values evwork   gsp014x;
label values wrkslf   gsp015x;
label values divorce  gsp016x;
label values spwrksta gsp017x;
label values ballot   gsp018x;


