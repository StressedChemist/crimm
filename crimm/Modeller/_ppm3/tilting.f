c     Modified for CRIMM on 2026-10-12; see NOTICE.md.
      subroutine tilting (phi,teta,nat,xyz,xyzc)
      common /ppm_error/ ppm_status
      integer ppm_status
c     ------------------------------------------
c     rotate coordinate set 'xyz' by angle 'teta' around axis in OXY
c     plane that forms phi anglr with OX axis
c     
      real xyz(3,*),xyzc(3,*),rot(3,3)
c
      a=sin(phi)
      b=cos(phi)
      si=sin(teta)
      co=cos(teta)
      co1=1.-co
      rot(1,1)=co+a*a*co1
      rot(1,2)=a*b*co1
      rot(1,3)=b*si
      rot(2,1)=rot(1,2)
      rot(2,2)=co+b*b*co1
      rot(2,3)=-a*si
      rot(3,1)=-rot(1,3)
      rot(3,2)=-rot(2,3)
      rot(3,3)=co
      do i=1,nat
         xyzc(1,i)=rot(1,1)*xyz(1,i)+rot(1,2)*xyz(2,i)+rot(1,3)*xyz(3,i)
         xyzc(2,i)=rot(2,1)*xyz(1,i)+rot(2,2)*xyz(2,i)+rot(2,3)*xyz(3,i)
         xyzc(3,i)=rot(3,1)*xyz(1,i)+rot(3,2)*xyz(2,i)+rot(3,3)*xyz(3,i)
      end do
      return
      end
c
      subroutine bundle_axis0 (nat,namat,xyz)
      common /ppm_error/ ppm_status
      integer ppm_status
c     --------------------------------------
c     Origin of coordinates corresponds
c     to center of mass of the set of CA atoms.
c
      real v(3),xyz(3,*)
c
      character*4 namat(*)
c
c     Define center of mass of CA atoms and
c     move all coordinates accordingly:
c
      do j=1,3
         v(j)=0.
      end do
c
      nca=0
      do i=1,nat
c**      if(namat(i).eq.'CA  ') then
            nca=nca+1
            do k=1,3
               v(k)=v(k)+xyz(k,i)
            end do
c**      end if
      end do
c
      do k=1,3
         v(k)=v(k)/float(nca)
         do i=1,nat
            xyz(k,i)=xyz(k,i)-v(k)
         end do
      end do
c
      return
      end
c
      subroutine bundle_axis (nat,xyz,xyzc,nsegm,isegm,
     *  natsegm,xyzca,iprint,numres,namsu,namat,namres)
      common /ppm_error/ ppm_status
      integer ppm_status
c     -------------------------------------------------------
c     Determine axes of all individual helices and axis of the entire
c     alpha-bundle.  Transform all coordinates to make this axis
c     coincide with Z axis.  Origin of coordinates corresponds
c     to center of mass of the set of CA atoms.
c
      parameter (maxsegm=900,maxlen=65,maxat=200000)
c
      common/params/ accs(maxat),iat(maxat),dip(maxat),eioniz(maxat),
     *  charge(maxat),asaref(maxat),hbond(maxat),charge2(maxat)
c
      real v(3),xyz(3,*),xyzca(3,maxsegm,*),xyzc(3,*),
     *  vt(3),vs(3,maxsegm),xyzbuf(3,maxlen),v1(3),v2(3)
c
      integer numres(*),natsegm(*),isegm(2,*)
c
      character*80 namfile
      character*1 namsu(*)
      character*4 namat(*),namres(*)
c
      data pi/3.14159/
c
c     Define center of mass of CA atoms and
c     move all coordinates accordingly:
c
      do j=1,3
         v(j)=0.
      end do
      nca=0
c
c     do i=1,nat
c        if(namat(i).eq.'CA  ') then
c           nca=nca+1
c           do k=1,3
c              v(k)=v(k)+xyz(k,i)
c           end do
c        end if
c     end do
c
c     do k=1,3
c        v(k)=v(k)/float(nca)
c        do i=1,nat
c           xyz(k,i)=xyz(k,i)-v(k)
c        end do
c     end do
c
      do i=1,nsegm
         nca=nca+natsegm(i)
         do j=1,natsegm(i)
            do k=1,3
               v(k)=v(k)+xyzca(k,i,j)
            end do
         end do
      end do
c 
      do k=1,3
         v(k)=v(k)/float(nca)
         do i=1,nsegm
            do j=1,natsegm(i)
               xyzca(k,i,j)=xyzca(k,i,j)-v(k)
            end do
         end do
         do i=1,nat
            xyz(k,i)=xyz(k,i)-v(k)
         end do
      end do
c
c     Determine axis of each helix:
c
      do i=1,nsegm
         do k=1,3
            v1(k)=0.
            v2(k)=0.
         end do
         nref=7
         l=isegm(2,i)-isegm(1,i)+1
         if(l.ge.7.and.l.le.15) nref=4
         if(l.ge.5.and.l.le.6) nref=3
         if(l.le.4) then
            do k=1,3
               vs(k,i)=xyzca(k,i,l)-xyzca(k,i,1)
            end do
         else
            do j=1,nref
               nn=natsegm(i)-nref+j
               do k=1,3
                  v1(k)=v1(k)+xyzca(k,i,j)
                  v2(k)=v2(k)+xyzca(k,i,nn)
               end do
            end do
            do k=1,3
               v1(k)=v1(k)/float(nref)   
               v2(k)=v2(k)/float(nref)   
               vs(k,i)=v2(k)-v1(k)
            end do
         end if  
         r=sqrt(vs(1,i)**2+vs(2,i)**2+vs(3,i)**2)
         do k=1,3
            vs(k,i)=vs(k,i)/r
         end do
      end do
c
c     Choose sign/directions of axis for each TM secondary
c     structure relative to OZ axis so that all SS vectors 
c     are facing roughtly in the same direction
c
      do i=1,nsegm
         r=sqrt(vs(1,i)*vs(1,i)+vs(2,i)*vs(2,i)+vs(3,i)*vs(3,i))
         if(r.lt.1.e-8) then
            ppm_status=6
            return
         end if
         ang=(acos(max(-1.,min(1.,vs(3,i)/r)))/pi)*180.
         if(ang.gt.90.) then
            do k=1,3
               vs(k,i)=-vs(k,i)
            end do
         end if
      end do
c
c     Vector of alpha-bundle axis:
c
      do k=1,3
         v(k)=0.
      end do
      do i=1,nsegm
         do k=1,3
            v(k)=v(k)+vs(k,i)
         end do
      end do
c
c     Angle betwen this axis and OZ:
c
      r=sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
      if(r.lt.1.e-8) then
         ppm_status=6
         return
      end if
      phi=acos(max(-1.,min(1.,v(3)/r)))
c     do k=1,3
c        v(k)=v(k)/r
c     end do
c
c     Perpendicular axis:
c
      v1(3)=0.
c     if((v(1).gt.0.and.v(2).gt.0.).or.(v(1).lt.0.and.v(2).lt.0.)) then
c        v1(1)=-v(2)
c        v1(2)= v(1)
c     else
         v1(1)= v(2)
         v1(2)=-v(1)
c     end if
c
c     Rotation:
c
      vt(1)=0.
      vt(2)=0.
      vt(3)=0.
      if(abs(v1(1))+abs(v1(2)).lt.1.e-8) then
         if(v(3).ge.0.) return
         v1(1)=1.
      end if
      call rotate (nat,xyz,phi,v1,vt) 
      do i=1,nsegm
         do j=1,natsegm(i)
            do k=1,3
               xyzbuf(k,j)=xyzca(k,i,j)
            end do
         end do
         call rotate (natsegm(i),xyzbuf,phi,v1,vt)
         do j=1,natsegm(i)
            do k=1,3
               xyzca(k,i,j)=xyzbuf(k,j)
            end do
         end do
      end do
      return
      end
c
      subroutine project (nat,xyz0)
      common /ppm_error/ ppm_status
      integer ppm_status
c     -----------------------------
c     Rotate the coordinate set to obtain the maximum
c     cross-section area in ZY plane, with the image tilted
c     in the right-upper direction
c
      parameter (maxat=200000)
c
      real xyz0(3,*),xyz(3,maxat)
      data pi/3.1459/
c
      smax=0.
      phimax=0.
      yzmax=0.
      do phi=0.,359.,1.
         ang=(phi/180.)*pi
         co=cos(ang)
         si=sin(ang)
         do i=1,nat
            xyz(1,i)=co*xyz0(1,i)-si*xyz0(2,i)
            xyz(2,i)=si*xyz0(1,i)+co*xyz0(2,i)
            xyz(3,i)=xyz0(3,i)
         end do
         yzmin=1000.
         ymin=1000.
         ymax=-1000.
         s=0.
         do zsl=-60.,60.,2. 
            zsl2=zsl+2.
            do i=1,nat
               if(xyz(3,i).gt.zsl.and.xyz(3,i).le.zsl2) then
                 if(xyz(1,i).lt.ymin) ymin=xyz(1,i)
                 if(xyz(1,i).gt.ymax) ymax=xyz(1,i)
               end if
            end do
            if(ymin.ne.1000.) then
               yzmax=ymax
               if(yzmin.eq.1000.) yzmin=ymin
               s=s+ymax-ymin
            end if
         end do
         if(s.gt.smax.and.yzmax.gt.yzmin) then
c        if(s.gt.smax) then
            smax=s
            phimax=phi
         end if
      end do
c
c     write (*,'(f7.1)') phimax
      ang=(phimax/180.)*pi
      co=cos(ang)
      si=sin(ang)
      do i=1,nat
         xyz(1,i)=co*xyz0(1,i)-si*xyz0(2,i)
         xyz(2,i)=si*xyz0(1,i)+co*xyz0(2,i)
         xyz(3,i)=xyz0(3,i)
      end do
      do i=1,nat
         xyz0(1,i)=xyz(1,i)
         xyz0(2,i)=xyz(2,i)
         xyz0(3,i)=xyz(3,i)
      end do
      return
      end
c
