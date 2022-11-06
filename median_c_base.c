// after sort2(a,b,temp): a>=b
#define sort2(a, b, temp) \
	(temp) = ((a)<(b))?(a):(b); \
	(a)=((a)>(b))?(a):(b); \
	(b)=(temp);

#define sort8(d0,d1,d2,d3,d4,d5,d6,d7,temp) \
	sort2(d0, d1, temp) \
	sort2(d2, d3, temp) \
	sort2(d4, d5, temp) \
	sort2(d6, d7, temp) \
	sort2(d0, d2, temp) \
	sort2(d1, d3, temp) \
	sort2(d4, d6, temp) \
	sort2(d5, d7, temp) \
	sort2(d1, d2, temp) \
	sort2(d5, d6, temp) \
	sort2(d0, d4, temp) \
	sort2(d1, d5, temp) \
	sort2(d2, d6, temp) \
	sort2(d3, d7, temp) \
	sort2(d2, d4, temp) \
	sort2(d3, d5, temp) \
	sort2(d1, d2, temp) \
	sort2(d3, d4, temp) \
	sort2(d5, d6, temp)


void main()
{
	register int r0=8, r1=4, r2=5, r3=1, r4=3, r5=7, r6=2, r7=6, r8;


	printf("Kezdeti allapot:\n");
	printf("%d", r0);
	printf("%d", r1);
	printf("%d", r2);
	printf("%d", r3);
	printf("%d", r4);
	printf("%d", r5);
	printf("%d", r6);
	printf("%d\n", r7);

	sort8(r0, r1, r2, r3, r4, r5, r6, r7, r8)
	printf("Vegallapot:\n");
	printf("%d", r0);
	printf("%d", r1);
	printf("%d", r2);
	printf("%d", r3);
	printf("%d", r4);
	printf("%d", r5);
	printf("%d", r6);
	printf("%d\n", r7);

}