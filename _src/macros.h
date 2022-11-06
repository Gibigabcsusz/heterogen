#pragma once

// after sort2(a,b,temp): a>=b
#define sort2(a, b, t) \
	(t) = ((a)<(b))?(a):(b); \
	(a)=((a)>(b))?(a):(b); \
	(b)=(t);

// after sort8(...): d0>=d1>=d2>=...>=d7
#define sort8(d0,d1,d2,d3,d4,d5,d6,d7,temp) \
	/*sort 4 pairs*/\
	sort2(d0, d1, temp) \
	sort2(d2, d3, temp) \
	sort2(d4, d5, temp) \
	sort2(d6, d7, temp) \
	/*merge 4 pairs into 2 groups of 4 sorted elements*/\
	sort2(d0, d2, temp) \
	sort2(d1, d3, temp) \
	sort2(d4, d6, temp) \
	sort2(d5, d7, temp) \
	sort2(d1, d2, temp) \
	sort2(d5, d6, temp) \
	/*merge groups of 4 to sorted group of 8*/\
	sort2(d0, d4, temp) \
	sort2(d1, d5, temp) \
	sort2(d2, d6, temp) \
	sort2(d3, d7, temp) \
	sort2(d2, d4, temp) \
	sort2(d3, d5, temp) \
	sort2(d1, d2, temp) \
	sort2(d3, d4, temp) \
	sort2(d5, d6, temp)

