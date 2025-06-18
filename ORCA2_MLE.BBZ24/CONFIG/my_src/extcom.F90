MODULE extcom
   !!======================================================================
   !!                       ***  MODULE  extcom  ***
   !! Demo external comm module: manage connexion with external codes 
   !!======================================================================
   !! History :  5.0.0  ! 2024-07  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   naminf          : machine learning models formulation namelist
   !!   extcom_init : initialization of Machine Learning based models
   !!   ext_comm        : ML based models
   !!   inf_snd         : send data to external trained model
   !!   inf_rcv         : receive data from external trained model
   !!----------------------------------------------------------------------
   USE oce             ! ocean fields
   USE dom_oce         ! ocean domain fields
   USE sbc_oce         ! ocean surface fields
   USE extfld          ! working fields for external models
   USE cpl_oasis3      ! OASIS3 coupling
   USE timing
   USE iom
   USE in_out_manager
   USE lib_mpp
   USE lbclnk

   IMPLICIT NONE
   PRIVATE

   PUBLIC extcom_alloc          ! function called in extcom_init 
   PUBLIC extcom_dealloc        ! function called in extcom_final
   PUBLIC extcom_init        ! routine called in nemogcm.F90
   PUBLIC ext_comm           ! routine called in stpmlf.F90
   PUBLIC extcom_final       ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_tmask = 1   ! t-grid mask
   INTEGER, PARAMETER ::   jps_gradb = 2   ! depth-averaged buoyancy gradient magnitude on t-grid
   INTEGER, PARAMETER ::   jps_fcor = 3    ! Coriolis parameter
   INTEGER, PARAMETER ::   jps_hml = 4     ! mixed-layer-depth on t-grid
   INTEGER, PARAMETER ::   jps_tau = 5     ! surface wind stress magnitude on t-grid
   INTEGER, PARAMETER ::   jps_q = 6       ! surface heat flux
   INTEGER, PARAMETER ::   jps_div = 7     ! depth-averaged horizontal divergence
   INTEGER, PARAMETER ::   jps_vort = 8    ! depth-averaged vertical vorticity
   INTEGER, PARAMETER ::   jps_strain = 9  ! depth-averaged strain magnitude
   INTEGER, PARAMETER ::   jps_ext = 9  ! total number of sendings

   INTEGER, PARAMETER ::   jpr_wb  = 1    ! depth-averaged subgrid vertical buoyancy flux on t-grid
   INTEGER, PARAMETER ::   jpr_ext = 1    ! total number of receptions

   INTEGER, PARAMETER ::   jpext = MAX(jps_ext,jpr_ext) ! Maximum number of exchanges

   TYPE( DYNARR ), SAVE, DIMENSION(jpext) ::  extsnd, extrcv  ! sent/received fields

   !!-------------------------------------------------------------------------
   !!                    Namelist for the Inference Models
   !!-------------------------------------------------------------------------
   LOGICAL , PUBLIC ::   ln_ext    !: activate module for inference models
   !!-------------------------------------------------------------------------

   !! Substitution
#  include "do_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OCE 5.0, NEMO Consortium (2024)
   !! Software governed by the CeCILL license (see ./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   INTEGER FUNCTION extcom_alloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION extcom_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpr_ext
         IF( srcv(nmodext)%fld(jn)%laction ) ALLOCATE( extrcv(jn)%z3(jpi,jpj,srcv(nmodext)%fld(jn)%nlvl), STAT=ierr )
         extcom_alloc = MAX(ierr,0)
      END DO     
      DO jn = 1, jps_ext
         IF( ssnd(nmodext)%fld(jn)%laction ) ALLOCATE( extsnd(jn)%z3(jpi,jpj,ssnd(nmodext)%fld(jn)%nlvl), STAT=ierr )
         extcom_alloc = MAX(ierr,0)
      END DO     
      !
   END FUNCTION extcom_alloc

   
   INTEGER FUNCTION extcom_dealloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION extcom_dealloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpr_ext
         IF( srcv(nmodext)%fld(jn)%laction ) DEALLOCATE( extrcv(jn)%z3, STAT=ierr )
         extcom_dealloc = MAX(ierr,0)
      END DO
      DO jn = 1, jps_ext
         IF( ssnd(nmodext)%fld(jn)%laction ) DEALLOCATE( extsnd(jn)%z3, STAT=ierr )
         extcom_dealloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION extcom_dealloc


   SUBROUTINE extcom_init 
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE extcom_init  ***
      !!
      !! ** Purpose :   Initialisation of the models that rely on external models
      !!
      !! ** Method  :   * Read naminf namelist
      !!                * create data for models
      !!----------------------------------------------------------------------
      !
      INTEGER ::   ios   ! Local Integer
      !!----------------------------------------------------------------------
      !
      ! ================================ !
      !      Namelist informations       !
      ! ================================ !
      !
      IF( lwp ) THEN                        ! control print
         WRITE(numout,*)
         WRITE(numout,*)'extcom_init : Setting external model'
         WRITE(numout,*)'~~~~~~~~~~~'
      END IF
      !
      IF( .NOT. lk_oasis )   CALL ctl_stop( 'extcom_init : External models coupled via OASIS, but key_oasis3 disabled' )
      !
      !
      ! ======================================== !
      !     Define exchange needs for Models     !
      ! ======================================== !
      !
      ALLOCATE( srcv(nmodext)%fld(jpr_ext) )
      !
      ! default definitions of ssnd snd srcv
      srcv(nmodext)%fld(:)%laction = .TRUE.  ;  srcv(nmodext)%fld(:)%clgrid = 'T'  ;  srcv(nmodext)%fld(:)%nsgn = 1.
      srcv(nmodext)%fld(:)%nct = 1  ;  srcv(nmodext)%fld(:)%nlvl = 1
      !
      ALLOCATE( ssnd(nmodext)%fld(jps_ext) )
      !
      ssnd(nmodext)%fld(:)%laction = .TRUE.  ;  ssnd(nmodext)%fld(:)%clgrid = 'T'  ;  ssnd(nmodext)%fld(:)%nsgn = 1.
      ssnd(nmodext)%fld(:)%nct = 1  ;  ssnd(nmodext)%fld(:)%nlvl = 1
      
      ! -------------------------------- !
      !          MLE-Fluxes-CNN          !
      ! -------------------------------- !
      ! sending gradb, FCOR, HML, TAU, Q, div, vort, strain
      ssnd(nmodext)%fld(jps_gradb)%clname = 'E_OUT_0'
      ssnd(nmodext)%fld(jps_fcor)%clname = 'E_OUT_1'
      ssnd(nmodext)%fld(jps_hml)%clname = 'E_OUT_2'
      ssnd(nmodext)%fld(jps_tau)%clname = 'E_OUT_3'
      ssnd(nmodext)%fld(jps_q)%clname = 'E_OUT_4'
      ssnd(nmodext)%fld(jps_div)%clname = 'E_OUT_5'
      ssnd(nmodext)%fld(jps_vort)%clname = 'E_OUT_6'
      ssnd(nmodext)%fld(jps_strain)%clname = 'E_OUT_7'
      ssnd(nmodext)%fld(jps_tmask)%clname = 'E_OUT_8'

      ! reception of vertical buoyancy fluxes
      srcv(nmodext)%fld(jpr_wb)%clname = 'E_IN_0'
      ! ------------------------------ !
      ! 
      ! ================================= !
      !   Define variables for coupling
      ! ================================= !
      CALL cpl_var(jpr_ext, jps_ext, 1, nmodext)
      !
      IF( extcom_alloc() /= 0 )  CALL ctl_stop( 'STOP', 'extcom_alloc : unable to allocate arrays' )
      IF( extfld_alloc() /= 0 )  CALL ctl_stop( 'STOP', 'extfld_alloc : unable to allocate arrays' ) 
      !
   END SUBROUTINE extcom_init


   SUBROUTINE ext_comm( kt, Kbb, Kmm, Kaa, hmld, bz, uz, vz )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE ext_comm  ***
      !!
      !! ** Purpose :   update the ocean data with the coupled models
      !!
      !! ** Method  :   *  
      !!                * 
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt            ! ocean time step
      INTEGER, INTENT(in) ::   Kbb, Kmm, Kaa ! ocean time level indices
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  hmld, bz, uz, vz
      !
      !
      INTEGER :: isec, info, jn                       ! local integer
      REAL(wp), DIMENSION(jpi,jpj)   ::  zdat, zdatx, zdaty   ! working buffer
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('ext_comm')
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange 
      info = OASIS_idle
      !
      ! ------  Prepare data to send ------
      !
      ! gradB
      CALL calc_2D_scal_gradient( bz, zdatx, zdaty )
      extsnd(jps_gradb)%z3(:,:,ssnd(nmodext)%fld(jps_gradb)%nlvl) = SQRT( zdatx(:,:)**2 + zdaty(:,:)**2 )
      ! FCOR
      extsnd(jps_fcor)%z3(:,:,ssnd(nmodext)%fld(jps_fcor)%nlvl) = ff_t(:,:)
      ! HML
      extsnd(jps_hml)%z3(:,:,ssnd(nmodext)%fld(jps_hml)%nlvl) = hmld(:,:)
      ! Tau
      !print*, jps_tau, size(ssnd(nmodext)%fld), size(taum), size(extsnd(jps_tau)%z3)
      extsnd(jps_tau)%z3(:,:,ssnd(nmodext)%fld(jps_tau)%nlvl) = taum(:,:)
      ! Heat Flux
      extsnd(jps_q)%z3(:,:,ssnd(nmodext)%fld(jps_q)%nlvl) = qsr(:,:) + qns(:,:)
      ! horizontal divergence
      CALL calc_2D_vec_hdiv( hmld, uz, vz, zdat )
      extsnd(jps_div)%z3(:,:,ssnd(nmodext)%fld(jps_div)%nlvl) = zdat(:,:)
      CALL iom_put( 'ext_div_mle', zdat)
      ! vorticity
      CALL calc_2D_vec_vort( uz, vz, zdat )
      extsnd(jps_vort)%z3(:,:,ssnd(nmodext)%fld(jps_vort)%nlvl) = zdat(:,:)
      CALL iom_put( 'ext_vort_mle', zdat)
      ! strain
      CALL calc_2D_strain_magnitude( uz, vz, zdat )
      extsnd(jps_strain)%z3(:,:,ssnd(nmodext)%fld(jps_strain)%nlvl) = zdat(:,:)
      CALL iom_put( 'ext_strain_mle', zdat)
      ! tmask
      extsnd(jps_tmask)%z3(:,:,1:ssnd(nmodext)%fld(jps_tmask)%nlvl) = tmask(:,:,1:ssnd(nmodext)%fld(jps_tmask)%nlvl)
      !
      ! ========================
      !   Proceed all sendings
      ! ========================
      !
      DO jn = 1, jps_ext
         IF ( ssnd(nmodext)%fld(jn)%laction ) THEN
            CALL cpl_snd( nmodext, jn, isec, extsnd(jn)%z3(A2D(0),:), info)
         ENDIF
      END DO
      !
      ! .... some external operations ....
      !
      ! ==========================
      !   Proceed all receptions
      ! ==========================
      !
      DO jn = 1, jpr_ext
         IF( srcv(nmodext)%fld(jn)%laction ) THEN
            CALL cpl_rcv( nmodext, jn, isec, extrcv(jn)%z3(A2D(0),:), info)
         ENDIF
      END DO
      !
      ! ------ Distribute receptions  ------
      !
      ! wb
      ext_wb(:,:) = extrcv(jpr_wb)%z3(:,:,srcv(nmodext)%fld(jpr_wb)%nlvl)
      !
      ! get streamfunction on correct grid points
      CALL invert_buoyancy_flux( ext_wb, zdatx, zdaty, bz,  ext_psiu, ext_psiv )
      !
      ! output results
      CALL iom_put( 'ext_wb', ext_wb )
      CALL iom_put( 'ext_psiu_mle', ext_psiu )
      CALL iom_put( 'ext_psiv_mle', ext_psiv )
      CALL iom_put( 'ext_bx_mle', zdatx )
      CALL iom_put( 'ext_by_mle', zdaty )
      CALL iom_put( 'ext_grdB_mle', SQRT( zdatx(:,:)**2 + zdaty(:,:)**2 ))
      CALL iom_put( 'ext_hmld_mle', hmld)
      CALL iom_put( 'ext_taum_mle', taum)
      CALL iom_put( 'ext_q_mle', qsr(:,:) + qns(:,:))
      CALL iom_put( 'ext_f_mle', ff_t)
      !
      IF( ln_timing )   CALL timing_stop('ext_comm')
      !
   END SUBROUTINE ext_comm

   SUBROUTINE invert_buoyancy_flux( wb, gradbx, gradby, scalar, psiu, psiv )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE invert_buoyancy_flux  ***
      !!
      !! ** Purpose :   Compute streamfunction on u- and v- points
      !!                from vertical buoyancy flux and buoyancy gradient
      !!
      !! ** Method  :   * Invert w'b' = psi x grad_b
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(inout) ::  wb, gradbx, gradby   ! vert. buoyncy flux
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  scalar  !  buoyancy
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) ::  psiu, psiv  ! computed streamfunction
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      INTEGER  :: jwgt              ! local storage integer
      REAL(wp) :: ampu, ampv
      REAL(wp), DIMENSION(jpi,jpj) :: dbu, dbv, dbu_v, dbv_u, wbu, wbv  ! buffers
      !!----------------------------------------------------------------------
      !
      ! invert buoyancy fluxes
      !
      ! interpolate to velocity points
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         dbu(ji,jj) = (scalar(ji+1,jj) - scalar(ji,jj))/e1u(ji,jj) * umask(ji,jj,1)
         dbv(ji,jj) = (scalar(ji,jj+1) - scalar(ji,jj))/e2v(ji,jj) * vmask(ji,jj,1)
      END_2D

      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         wbu(ji,jj) = 0.5*(wb(ji+1,jj) + wb(ji,jj)) * umask(ji,jj,1)
         wbv(ji,jj) = 0.5*(wb(ji,jj+1) + wb(ji,jj)) * vmask(ji,jj,1)
      END_2D

      CALL lbc_lnk( 'infmod', gradbx, 'T', 1.0_wp , gradby, 'T', 1.0_wp )
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         jwgt = tmask(ji,jj+1,1) + tmask(ji,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         dbu_v(ji,jj) = ( gradbx(ji,jj+1) + gradbx(ji,jj) ) / REAL(jwgt,wp)

         jwgt = tmask(ji+1,jj,1) + tmask(ji,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         dbv_u(ji,jj) = ( gradby(ji+1,jj) + gradby(ji,jj) ) / REAL(jwgt,wp)
      END_2D

      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         ampu = dbu(ji,jj)**2+dbv_u(ji,jj)**2
         IF ( ampu == 0.0_wp ) ampu = 1.0_wp
         ampv = dbv(ji,jj)**2+dbu_v(ji,jj)**2
         IF ( ampv == 0.0_wp ) ampv = 1.0_wp
         psiu(ji,jj) = ( wbu(ji,jj)/ MAX( ampu, (wbu(ji,jj) / 1E2)**2) ) * dbu(ji,jj) * umask(ji,jj,1)
         psiv(ji,jj) = ( wbv(ji,jj)/ MAX( ampv, (wbv(ji,jj) / 1E2)**2) ) * dbv(ji,jj) * vmask(ji,jj,1)
      END_2D
      !
   END SUBROUTINE invert_buoyancy_flux


   SUBROUTINE calc_2D_scal_gradient( scalar, gradx, grady )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_scal_gradient  ***
      !!
      !! ** Purpose :   Compute gradient of a 2D scalar field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  scalar        ! input scalar
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: gradx, grady  ! computed gradient
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      INTEGER  :: jwgt              ! local storage integer
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         ! grad in i-longitude
         gradx(ji,jj) = ( scalar(ji+1,jj) - scalar(ji,jj) ) / e1u(ji,jj) * umask(ji,jj,1)
         gradx(ji,jj) = gradx(ji,jj) + ( scalar(ji,jj) - scalar(ji-1,jj) ) / e1u(ji-1,jj) * umask(ji-1,jj,1)
         jwgt = umask(ji,jj,1) + umask(ji-1,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         gradx(ji,jj) = gradx(ji,jj) / REAL(jwgt,wp)
        
         ! grad in j-latitude
         grady(ji,jj) = ( scalar(ji,jj+1) - scalar(ji,jj) ) / e2v(ji,jj) * vmask(ji,jj,1)
         grady(ji,jj) = grady(ji,jj) + ( scalar(ji,jj) - scalar(ji,jj-1) ) / e2v(ji,jj-1) * vmask(ji,jj-1,1)
         jwgt = vmask(ji,jj,1) + vmask(ji,jj-1,1)
         IF ( jwgt == 0 ) jwgt = 1
         grady(ji,jj) = grady(ji,jj) / REAL(jwgt,wp)
      END_2D
      !
   END SUBROUTINE calc_2D_scal_gradient


   SUBROUTINE calc_2D_vec_hdiv( hgt, u, v, hdiv )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_vec_hdiv  ***
      !!
      !! ** Purpose :   Compute horizontal divergence of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  hgt     ! thickness of 2D field
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: hdiv   ! computed divergence
      !
      INTEGER  ::   ji, jj       ! dummy loop indices
      REAL(wp)  :: ztmp1, ztmp2  ! interpolated hgt in u- and v- points
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls-1, nn_hls, nn_hls-1, nn_hls )
         ! i-longitude
         ztmp1 = MIN( hgt(ji+1,jj) , hgt(ji,jj) )
         ztmp2 = MIN( hgt(ji,jj) , hgt(ji-1,jj) )
         hdiv(ji,jj) = u(ji,jj) * e2u(ji,jj) * ztmp1 - u(ji-1,jj) * e2u(ji-1,jj) * ztmp2
         
         ! j-latitude
         ztmp1 = MIN( hgt(ji,jj+1) , hgt(ji,jj) )
         ztmp2 = MIN( hgt(ji,jj) , hgt(ji,jj-1) )
         hdiv(ji,jj) = hdiv(ji,jj) + ( v(ji,jj) * e1v(ji,jj) * ztmp1 - v(ji,jj-1) * e1v(ji,jj-1) * ztmp2 )

         hdiv(ji,jj) = hdiv(ji,jj)  / ( e1e2t(ji,jj)*hgt(ji,jj) )
      END_2D
      !
   END SUBROUTINE calc_2D_vec_hdiv


   SUBROUTINE calc_2D_vec_vort( u, v, vort )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_vec_vort  ***
      !!
      !! ** Purpose :   Compute vertical vorticity of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: vort     ! computed vorticiy
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      REAL(wp), DIMENSION(jpi,jpj) :: zbuf ! working buffer
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         zbuf(ji,jj) = e2v(ji+1,jj) * v(ji+1,jj) - e2v(ji,jj) * v(ji,jj)
         zbuf(ji,jj) = zbuf(ji,jj) - e1u(ji,jj+1) * u(ji,jj+1) + e1u(ji,jj) * u(ji,jj)
         zbuf(ji,jj) = zbuf(ji,jj) * r1_e1e2f(ji,jj) * fmask(ji,jj,1)
      END_2D
      !
      ! set on t-grid
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         vort(ji,jj) = 0.25_wp * ( zbuf(ji-1,jj) + zbuf(ji,jj) + zbuf(ji-1,jj-1) + zbuf(ji,jj-1) )
      END_2D
      !
   END SUBROUTINE calc_2D_vec_vort


   SUBROUTINE calc_2D_strain_magnitude( u, v, strain )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_strain_magnitude  ***
      !!
      !! ** Purpose :   Compute strain magnitude of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: strain ! computed strain
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      REAL(wp)  :: ztmp             ! local real
      REAL(wp), DIMENSION(jpi,jpj) :: ztrac, zshear ! working arrays
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         ! expansion rate
         ztmp =   ( u(ji,jj)*r1_e2u(ji,jj) - u(ji-1,jj)*r1_e2u(ji-1,jj) ) * r1_e1t(ji,jj) * e2t(ji,jj) &
              & - ( v(ji,jj)*r1_e1v(ji,jj) - v(ji,jj-1)*r1_e1v(ji,jj-1) ) * r1_e2t(ji,jj) * e1t(ji,jj)  
         ztrac(ji,jj) = ztmp**2 * tmask(ji,jj,1)

         ! shear rate 
         ztmp =   ( u(ji,jj+1)*r1_e1u(ji,jj+1) - u(ji,jj)*r1_e1u(ji,jj) ) * r1_e2f(ji,jj) * e1f(ji,jj) &
              & + ( v(ji+1,jj)*r1_e2v(ji+1,jj) - v(ji,jj)*r1_e2v(ji,jj) ) * r1_e1f(ji,jj) * e2f(ji,jj) 
         zshear(ji,jj) = ztmp**2 * fmask(ji,jj,1)
      END_2D
      !
      ! t-grid
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         strain(ji,jj) = 0.25_wp * ( zshear(ji-1,jj) + zshear(ji,jj) + zshear(ji-1,jj-1) + zshear(ji,jj-1) )
         strain(ji,jj) = SQRT( strain(ji,jj) + ztrac(ji,jj) ) 
      END_2D
      !
   END SUBROUTINE calc_2D_strain_magnitude

   SUBROUTINE extcom_final
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE extcom_final  ***
      !!
      !! ** Purpose :   Free memory used for extcom modules
      !!
      !! ** Method  :   * Deallocate arrays
      !!----------------------------------------------------------------------
      !
      IF( extcom_dealloc() /= 0 )     CALL ctl_stop( 'STOP', 'extcom_dealloc : unable to free memory' )
      IF( extfld_dealloc() /= 0 )  CALL ctl_stop( 'STOP', 'extfld_dealloc : unable to free memory' )      
      !
   END SUBROUTINE extcom_final 
   !!=======================================================================
END MODULE extcom
