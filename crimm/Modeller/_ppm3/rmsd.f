c     Modified for CRIMM on 2026-10-12; see NOTICE.md.
      subroutine rmsd (nat,xref,xmod,v1,v2,rot,rms)
      common /ppm_error/ ppm_status
      integer ppm_status
c     -----------------------------------------
c     superposition of two coordinate sets:
c     xref - "reference" (experimental) coordinates
c     xmod - model coordinates that are transformed
c     to be superimposed with xref
c
      parameter (maxat=200000)
c
      real xref(3,*),xmod(3,*),v1(3),v2(3),xc(maxat),yc(maxat),
     *  zc(maxat),ac(maxat),bc(maxat),cc(maxat),rot(3,3)
c
      do j=1,3
         v1(j)=0.
      end do
c
      do i=1,nat
         do j=1,3
            v1(j)=v1(j)+xref(j,i)
         end do
      end do
c
      do j=1,3
         v1(j)=v1(j)/float(nat)
      end do
c
      do j=1,3
         v2(j)=0.
      end do
c
      do k=1,nat
         do j=1,3
            v2(j)=v2(j)+xmod(j,k)
         end do
      end do
      do j=1,3
         v2(j)=v2(j)/float(nat)
      end do
c
      do k=1,nat
         ac(k)=xref(1,k)-v1(1)
         bc(k)=xref(2,k)-v1(2)
         cc(k)=xref(3,k)-v1(3)
         xc(k)=xmod(1,k)-v2(1)
         yc(k)=xmod(2,k)-v2(2)
         zc(k)=xmod(3,k)-v2(3)
c        write(*,'(6f8.3)') xc(k),yc(k),zc(k),ac(k),bc(k),cc(k)
      end do
c
      call cmpcor (nat,xc,yc,zc,ac,bc,cc,0,rms,rot,ier)
c
      if(ier.ne.0) ppm_status=6
c
c     do i=1,nat
c        a=xc(i)*rot(1,1)+yc(i)*rot(1,2)+zc(i)*rot(1,3)
c        b=xc(i)*rot(2,1)+yc(i)*rot(2,2)+zc(i)*rot(2,3)
c        c=xc(i)*rot(3,1)+yc(i)*rot(3,2)+zc(i)*rot(3,3)
c        xmod(1,i)=a + v1(1)-v2(1)
c        xmod(2,i)=b + v1(2)-v2(2)
c        xmod(3,i)=c + v1(3)-v2(3)
c     end do
c     if(rms.gt.0.01) write(*,'(''rmsd='',f6.2,i4)') rms,nat
c
      return
      end
c
c     cmp system***15.12.86***
c     subroutine cmpcor to clc rmsd between structures
c     xyz and abc (mclachlan a.d., jmb, 1979, v.128, p.49-79)
c
c     n - no.of points
c     mv=1 - without rot matrix clc, =0 with one
c     xc,yc,zc and ac,bc,cc  - comparing coordinate' sets
c     rmsd - value of difference between the sets
c     rot  - rotation matrix for best coincidence a and b
c            to rotate xyz set to abc set :
c
c            ac'(i)=xc(i)*rot(1,1)+yc(i)*rot(1,2)+zc(i)*rot(1,3)
c            bc'(i)=xc(i)*rot(2,1)+yc(i)*rot(2,2)+zc(i)*rot(2,3)
c            cc'(i)=xc(i)*rot(3,1)+yc(i)*rot(3,2)+zc(i)*rot(3,3)
c
c     ier = 0 - unique rot matrix
c         = 1 - rot matrix has one degree  of freedom
c         = 2 - rot matrix has two degrees of freedom
c         =-1 - best fit is undefined
c         =-2 - sqrt from negative rmsd squared
c
c     subunits: eigen
c     files: none
c     funct.lib.: sqrt(real), abs(real)
c
      subroutine cmpcor(n,xc,yc,zc,ac,bc,cc,mv, rmsd,rot,ier)
      common /ppm_error/ ppm_status
      integer ppm_status
c
      dimension xc(*),yc(*),zc(*),ac(*),bc(*),cc(*)
      dimension rot(3,3),u(3,3),omega(6,6),omgl(21),eigom(36),
     *          evecth(3,3),evectk(3,3),r1(3),r2(3),sgn(3)
c
      data epsi/1.0e-04/,twsqrt/1.41421356/
c
      ier=0
      rmsd=0.0
c
      fdiff=0.0
      do 10 i=1,n
        fdiff=fdiff+xc(i)*xc(i)+ac(i)*ac(i)+yc(i)*yc(i)+bc(i)*bc(i)+
     *              zc(i)*zc(i)+cc(i)*cc(i)
  10  continue
      fdiff=fdiff/(2*n)
c
      do 12 k=1,3
        do 11 l=1,3
          u(k,l)=0.0
  11    continue
  12  continue
c
      dn=1.0/n
      do 15 i=1,n
        r1(1)=xc(i)
        r1(2)=yc(i)
        r1(3)=zc(i)
        r2(1)=ac(i)
        r2(2)=bc(i)
        r2(3)=cc(i)
        do 14 k=1,3
          do 13 l=1,3
            u(k,l)=u(k,l)+r1(k)*r2(l)*dn
  13      continue
  14    continue
  15  continue
      detu=u(1,1)*(u(2,2)*u(3,3)-u(2,3)*u(3,2))+
     1     u(1,2)*(u(2,3)*u(3,1)-u(2,1)*u(3,3))+
     2     u(1,3)*(u(2,1)*u(3,2)-u(2,2)*u(3,1))
c
      if(detu.ne.0.0)goto 20
         ier=-1
         goto 100
c
  20  sgndu=detu/abs(detu)
      do 22 i=1,6
        do 21 j=1,6
          omega(i,j)=0.0
  21    continue
  22  continue
      do 24 k=1,3
        do 23 l=1,3
          omega(k,l+3)=u(k,l)
  23    continue
  24  continue
      do 26 j=1,6
        do 25 i=1,j
          ij=i+j*(j-1)/2
          omgl(ij)=omega(i,j)
  25    continue
  26  continue
      call eigen(omgl,eigom,6,mv)
      rlamb1=omgl(1)
      rlamb2=omgl(3)
      rlamb3=omgl(6)
c
      if(detu.ge.0.0)goto 32
         if(rlamb2.ne.rlamb3)goto 32
            if(rlamb1.ne.rlamb2)goto 31
               ier=2
               slamb=rlamb1
               goto 33
  31  ier=1
      slamb=rlamb1
      goto 33
c
  32  slamb=rlamb1+rlamb2+sgndu*rlamb3
  33  rmsd=fdiff-slamb
      if(rmsd.ge.0.0)goto 35
         if(rmsd+epsi.lt.0.0)goto 34
            rmsd=0.0
            goto 35
  34  ier=-2
      goto 100
c
  35  rmsd=sqrt(2.0*rmsd)
c
      if(mv.eq.1)goto 100
      do 42 k=1,3
        do 41 l=1,3
          klh=6*(k-1)+l
          klk=klh+3
          evecth(k,l)=twsqrt*eigom(klh)
          evectk(k,l)=twsqrt*eigom(klk)
  41    continue
  42  continue
c
      evecth(3,1)=evecth(1,2)*evecth(2,3)-evecth(1,3)*evecth(2,2)
      evecth(3,2)=evecth(1,3)*evecth(2,1)-evecth(1,1)*evecth(2,3)
      evecth(3,3)=evecth(1,1)*evecth(2,2)-evecth(1,2)*evecth(2,1)
c
c     change sign in k-vector if det u-matrix< 0
c
      qf=1.
      if(sgndu.lt.0.)qf=-1.
c
      evectk(3,1)=qf*(evectk(1,2)*evectk(2,3)-evectk(1,3)*evectk(2,2))
      evectk(3,2)=qf*(evectk(1,3)*evectk(2,1)-evectk(1,1)*evectk(2,3))
      evectk(3,3)=qf*(evectk(1,1)*evectk(2,2)-evectk(1,2)*evectk(2,1))
c
      sgn(1)=1.0
      sgn(2)=1.0
      sgn(3)=sgndu
c
      do k=1,3
        do l=1,3
          rot(k,l)=0.
        end do
      end do
      do 45 k=1,3
        do 44 l=1,3
          rot(k,l)=0.0
          do 43 m=1,3
            rot(k,l)=rot(k,l)+evectk(m,k)*evecth(m,l)*sgn(m)
  43      continue
  44    continue
  45  continue
c
 100  return
      end
c
      subroutine eigen(a,r,n,mv)
      common /ppm_error/ ppm_status
      integer ppm_status
c     --------------------------
c     general***15.12.86***  from ssp-1966
c !!! here a and r dimensioned as 21 and 36, respectively
c
      dimension a(21),r(36)
c
c     if(mv-1)10,25,10
      if(mv-1.eq.0) go to 25
      iq=-n
      do 21 j=1,n
      iq=iq+n
      do 20 i=1,n
      ij=iq+i
      r(ij)=0.0
c     if(i-j)20,15,20
      if(i-j.ne.0) go to 20
      r(ij)=1.0
20    continue
21    continue
c
25    anorm=0.0
      do 35 i=1,n
      do 34 j=1,n
c     if(i-j)30,35,30
      if(i-j.eq.0) go to 34
      ia=i+(j*j-j)/2
      anorm=anorm+a(ia)*a(ia)
34    continue
35    continue
c     if(anorm)165,165,40
      if(anorm.le.0.) go to 165
      anorm=1.414*sqrt(anorm)
      anrmx=anorm*1.0e-10/float(n)
c
      ind=0
      thr=anorm
45    thr=thr/float(n)
50    l=1
55    m=l+1
c
60    mq=(m*m-m)/2
      lq=(l*l-l)/2
      lm=l+mq
  62  continue
c     if(abs(a(lm))-thr)130,65,65
      if(abs(a(lm))-thr.lt.0.) go to 130
      ind=1
      ll=l+lq
      mm=m+mq
      x=0.5*(a(ll)-a(mm))
68    y=-a(lm)/sqrt(a(lm)*a(lm)+x*x)
c     if(x)70,75,75
      if(x.ge.0.) go to 75
      y=-y
75    sinx=y/sqrt(2.0*(1.0+(sqrt(1.0-y*y))))
      sinx2=sinx*sinx
78    cosx=sqrt(1.0-sinx2)
      cosx2=cosx*cosx
      sincs=sinx*cosx
c
c
      ilq=n*(l-1)
      imq=n*(m-1)
      do 125 i=1,n
      iq=(i*i-i)/2
c     if(i-l)80,115,80
      if(i-l.eq.0) go to 115
   80 continue
c80    if(i-m)85,115,90
      if(i-m.eq.0) go to 115
      if(i-m.gt.0) go to 90
      im=i+mq
      goto 95
90    im=m+iq
   95 continue
c95    if(i-l)100,105,105
      if(i-l.ge.0) go to 105
100   il=i+lq
      goto 110
105   il=l+iq
110   x=a(il)*cosx-a(im)*sinx
      a(im)=a(il)*sinx+a(im)*cosx
      a(il)=x
  115 continue
c115   if(mv-1)120,125,120
      if(mv-1.eq.0) go to 125
      ilr=ilq+i
      imr=imq+i
      x=r(ilr)*cosx-r(imr)*sinx
      r(imr)=r(ilr)*sinx+r(imr)*cosx
      r(ilr)=x
125   continue
      x=2.0*a(lm)*sincs
      y=a(ll)*cosx2+a(mm)*sinx2-x
      x=a(ll)*sinx2+a(mm)*cosx2+x
      a(lm)=(a(ll)-a(mm))*sincs+a(lm)*(cosx2-sinx2)
      a(ll)=y
      a(mm)=x
c
  130 continue
      if(m-n.eq.0) go to 140
c130   if(m-n)135,140,135
135   m=m+1
      goto 60
c
  140 continue
c140   if(l-(n-1))145,150,145
      if(l-(n-1).eq.0) go to 150
      l=l+1
      goto 55
150   if(ind-1.ne.0) go to 160
c150   if(ind-1)160,155,160
      ind=0
      goto 50
c
160   if(thr-anrmx.gt.0.) go to 45
c160   if(thr-anrmx)165,165,45
c
165   iq=-n
c
      do 186 i=1,n
      iq=iq+n
      ll=i+(i*i-i)/2
      jq=n*(i-2)
      do 185 j=i,n
      jq=jq+n
      mm=j+(j*j-j)/2
c     if(a(ll)-a(mm))170,185,185
      if(a(ll)-a(mm).ge.0.) go to 185
      x=a(ll)
      a(ll)=a(mm)
      a(mm)=x
c     if(mv-1)175,185,175
      if(mv-1.eq.0) go to 185
      do k=1,n
         ilr=iq+k
         imr=jq+k
         x=r(ilr)
         r(ilr)=r(imr)
         r(imr)=x
      end do
185   continue
186   continue
      return
      end
