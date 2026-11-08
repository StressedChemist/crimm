c     Modified for CRIMM on 2026-10-12; see NOTICE.md.
c     Planar array driver for the pinned PPM3 positioning routines.
      subroutine ppm3_driver(nat,xyz,numres,namsu,namat,namres,
     *  reslib,ph,temp,sysinp,rotation,translation,metrics,matching)
      parameter (maxat=200000,maxres=30000,maxlen=65,maxsegm=900)
      common /ppm_error/ ppm_status
      integer ppm_status
      character*(*) reslib
      character*3 sysinp
      character*80 pdbtempl
      character*1 namsu(*),namsegm(maxsegm)
      character*4 namat(*),namres(*)
      integer numres(*),ienvat(maxat),natsegm(maxsegm),
     *  ifirst(maxres),ilast(maxres),ienv(maxres),isegm(2,maxsegm)
      real xyz(3,*),solv(maxat),xyzbuf(3,10),
     *  xyzc(3,maxat),xyzca(3,maxsegm,maxlen),vcenter(3),
     *  rotation(3,3),translation(3),metrics(6),matching(3),v1(3),v2(3)
      data pi/3.14159/
c
      pdbtempl='CRIMM'
      iprint=0
      nlipid=0
      call prepare(nat,xyz,solv,numres,namsu,iprint,namat,
     *  namres,nres,ifirst,ilast,reslib,ph,temp)
      if(ppm_status.ne.0) return
      nsuper=min(nat,10)
      xyzbuf(:,1:nsuper)=xyz(:,1:nsuper)
c
      call orient(nat,xyz,solv,phimax,tetamax,vcenter,
     *  dmax,zcenter,emin)
      if(ppm_status.ne.0) return
      do i=1,nat
         xyz(:,i)=xyz(:,i)-vcenter
      end do
      phimax=phimax*pi/180.
      tetamax=tetamax*pi/180.
      call tilting(phimax,tetamax,nat,xyz,xyzc)
      do i=1,nat
         xyz(1,i)=xyzc(1,i)
         xyz(2,i)=xyzc(2,i)
         xyz(3,i)=xyzc(3,i)-zcenter
      end do
c
      nsegm=0
      call find_segm(nat,xyz,xyzca,numres,namat,namsu,
     *  namres,nsegm,isegm,namsegm,natsegm,dmax,itypess,0,0.)
      if(ppm_status.ne.0) return
      model=1
      if(nsegm.ge.1) model=2
      if(dmax.gt.18..and.nsegm.eq.0) model=3
      call ppm3_parameters(sysinp,model,itypess,matching)
      if(ppm_status.ne.0) return
      dmatch=matching(2)
      emism=matching(3)
      if(model.eq.1.or.model.eq.3) then
         call bundle_axis0(nat,namat,xyz)
         call watface0(nat,ienvat,nres,ienv)
      else
         call bundle_axis(nat,xyz,xyzc,nsegm,isegm,natsegm,
     *     xyzca,iprint,numres,namsu,namat,namres)
         if(ppm_status.ne.0) return
         call watface(nat,xyz,numres,namsu,nres,ifirst,ilast,
     *     ienv,iprint,pdbtempl,nsegm,natsegm,xyzca,model,
     *     ienvat,namat,namres,nlipid)
      end if
c
      ifunc=2
      call optim(model,dmin,emism,dmatch,nsegm,nat,xyz,xyzc,
     *  solv,numres,namsu,shift,tilt,ener,iprint,namat,namres,
     *  pdbtempl,ifunc,d12,t12,nlipid,sysinp)
      if(ppm_status.ne.0) return
c     Thickness scans do not transform atoms; retain the final locate.
      call profile(dmin)
      call locate(nat,xyz,xyzc,solv,numres,namsu,dmin,
     *  shift,tilt,ener,iprint,namat,namres,
     *  pdbtempl,ifunc,s12,t12,1)
      if(ppm_status.ne.0) return
      ener=ener+float(nlipid)*emism*(dmin-dmatch)**2
      call project(nat,xyz)
      call rmsd(nsuper,xyz,xyzbuf,v1,v2,rotation,rms)
      if(ppm_status.ne.0) return
      translation=v1-matmul(rotation,v2)
      metrics(1)=ener
      metrics(2)=dmin
      metrics(3)=tilt
      metrics(4)=d12
      metrics(5)=t12
      metrics(6)=float(model)
      end
c
      subroutine ppm3_parameters(sysinp,model,itypess,matching)
      common /ppm_error/ ppm_status
      integer ppm_status
      character*3 sysinp,sysname(21)/'PMm','PMp','PMf','ERf',
     *  'ERm','GOL','LYS','END','VAC','MOM','MIM','THp','THb','GnO',
     *  'GnI','GpI','ARC','LPC','MPC','OPC','EPC'/
      real matching(3),deq(21)/32.0,2*30.6,2*28.7,30.0,
     *  30.7,30.0,30.4,22.7,28.4,30.2,30.8,23.9,30.1,30.3,
     *  30.2,21.7,25.7,28.8,35.8/,
     *  eeq(21)/9*0.02,2*0.01,2*0.02,4*0.015,4*0.02/
      do i=1,21
         if(sysinp.eq.sysname(i)) then
            matching(1)=deq(i)
            matching(2)=deq(i)
            matching(3)=eeq(i)
            if(model.eq.2.and.itypess.eq.2.and.sysinp.ne.'MOM'
     *          .and.sysinp.ne.'GnO') then
               matching(2)=23.9
               matching(3)=0.01
            end if
            return
         end if
      end do
      ppm_status=1
      end
c
      subroutine optim (model,dmin,emism,dmatch,nsegm,nat,xyz,
     *  xyzc,solv,numres,namsu,shift,tilt,ener,iprint,namat,namres,
     *  pdbtempl,ifunc,d12,t12,nlipid,sysinp)
      common /ppm_error/ ppm_status
      integer ppm_status
c     ------------------------------------------------------------
c     Minimize transfer energy. Free variables for optimization:
c     membrane thickness for TM version; shift of center
c     of mass along Z axis; and rotation angles arond OX and OY axes.
c
      character*80 pdbtempl
      character*1 namsu(*)
      character*3 sysinp
      character*4 namat(*),namres(*)
      integer numres(*)
      real xyz(3,*),solv(*),xyzc(3,*),dtab(500),etab(500)
c
      data ecut_tab/1.0/
c     data ecut_tab/1.0/,emism/0.0/
c
      if(model.eq.1) then
         dmembr=dmatch
         call profile (dmembr)
         call locate (nat,xyz,xyzc,solv,numres,namsu,dmembr,
     *     shift,tilt,ener,iprint,namat,namres,
     *     pdbtempl,ifunc,s12,t12,1)
         if(ppm_status.ne.0) return
         dmin=dmembr
         d12=s12
      else
         emin=99.
         dmin=99.
         m=0
         dm1=dmatch-3.2
         dm2=dmatch+3.2
         if(sysinp.eq.'   ') then
            dm1=dmatch-5.0
            dm2=dmatch+5.0
         end if
         do dmembr=dm1,dm2,0.2
            call profile (dmembr)
            call locate (nat,xyz,xyzc,solv,numres,namsu,dmembr,
     *        shift,tilt,ener,iprint,namat,namres,
     *        pdbtempl,ifunc,s12,t12,0)
            if(ppm_status.ne.0) return
c           diff=abs(dmembr-dmatch)
c           if(diff.gt.0.5) then
c              dmism=diff-0.5
c           else
c              dmism=0.
c           end if
c           ener=ener+float(nlipid)*emism*dmism
            ener=ener+float(nlipid)*emism*(dmembr-dmatch)**2
            m=m+1
            etab(m)=ener
            dtab(m)=dmembr
c           write (*,'(2f9.3,f8.1)') dtab(m),etab(m),tilt
            if(ener.lt.emin) then
               emin=ener
               dmin=dmembr
            end if
         end do
         if(dmin.eq.99.) then
            ppm_status=5
            return
         end if
         ntab=m
         dmin0=100.
         dmax0=0.
         do i=1,ntab
c           write (*,'(2f9.3)') dtab(i),etab(i)
            if(etab(i).le.emin+ecut_tab) then
               if(dtab(i).lt.dmin0) dmin0=dtab(i)
               if(dtab(i).gt.dmax0) dmax0=dtab(i)
            end if
         end do
         d12=0.5*(dmax0-dmin0)
c        call profile (dmin)
c        call locate (nat,xyz,xyzc,solv,numres,namsu,dmin,
c    *     shift,tilt,ener,iprint,namat,namres,
c    *     pdbtempl,ifunc,s12,t12,1)
c           diff=abs(dmin-dmatch)
c           if(diff.gt.0.5) then
c              dmism=diff-0.5
c           else
c              dmism=0.
c           end if
c          ener=ener+float(nlipid)*emism*dmism
         ener=emin
      end if
      return
      end
c
