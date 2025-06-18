MODULE extfld
   !!======================================================================
   !!                       ***  MODULE extfld  ***
   !! Inferences module :   variables defined in core memory
   !!======================================================================
   !! History :  4.2  ! 2023-09  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   extfld_alloc : allocation of fields arrays for inferences module (infmod)
   !!----------------------------------------------------------------------        
   !!=====================================================
   USE par_oce        ! ocean parameters
   USE lib_mpp        ! MPP library

   IMPLICIT NONE
   PRIVATE

   PUBLIC   extfld_alloc   ! routine called in infmod.F90
   PUBLIC   extfld_dealloc ! routine called in infmod.F90

   !!----------------------------------------------------------------------
   !!                    2D Inference Module fields
   !!----------------------------------------------------------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)  :: ext_psiu, ext_psiv   !: external-computed MLE streamfunction
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)  :: ext_wb               !: external-computed vertical buoyancy fluxes

   !!----------------------------------------------------------------------
   !!                    3D Inference Module fields
   !!----------------------------------------------------------------------

CONTAINS

   INTEGER FUNCTION extfld_alloc()
      !!---------------------------------------------------------------------
      !!                  ***  FUNCTION extfld_alloc  ***
      !!---------------------------------------------------------------------
      INTEGER :: ierr
      !!---------------------------------------------------------------------
      ierr = 0
      !
      ALLOCATE( ext_psiu(jpi,jpj) , ext_psiv(jpi,jpj) , ext_wb(jpi,jpj) , STAT=ierr )
      extfld_alloc = ierr
      !
   END FUNCTION

   
   INTEGER FUNCTION extfld_dealloc()
      !!---------------------------------------------------------------------
      !!                  ***  FUNCTION extfld_dealloc  ***
      !!---------------------------------------------------------------------
      INTEGER :: ierr
      !!---------------------------------------------------------------------
      ierr = 0
      !
      DEALLOCATE( ext_psiu, ext_psiv, ext_wb , STAT=ierr )
      extfld_dealloc = ierr
      !
   END FUNCTION

END MODULE extfld
