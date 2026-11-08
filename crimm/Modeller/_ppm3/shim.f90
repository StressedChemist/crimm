! Added for CRIMM on 2026-10-12; see NOTICE.md.
! The C ABI is double precision. Upstream calculations use default REAL.
subroutine ppm3_orient(n, input_xyz, atom_bytes, residue_bytes, chain_bytes, &
                      residue_ids, path_bytes, path_length, ph, temperature, &
                      topology, system_bytes, rotation, translation, metrics, matching, status) &
                      bind(C, name='ppm3_orient')
  use iso_c_binding, only: c_int, c_double, c_char
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use, intrinsic :: ieee_exceptions, only: ieee_get_flag, ieee_set_flag, ieee_invalid, ieee_divide_by_zero
  implicit none
  integer(c_int), value :: n, path_length, topology
  real(c_double), value :: ph, temperature
  real(c_double), intent(in) :: input_xyz(3,*)
  character(c_char), intent(in) :: atom_bytes(*), residue_bytes(*), chain_bytes(*)
  character(c_char), intent(in) :: path_bytes(*), system_bytes(3)
  integer(c_int), intent(in) :: residue_ids(*)
  real(c_double), intent(out) :: rotation(3,3), translation(3), metrics(6), matching(3)
  integer(c_int), intent(out) :: status
  integer :: ppm_status, i, j, alloc_status
  common /ppm_error/ ppm_status
  real, allocatable :: xyz(:,:)
  character(4), allocatable :: atom_names(:), residue_names(:)
  character(1), allocatable :: chains(:)
  integer, allocatable :: ids(:)
  character(4096) :: residue_library
  character(3) :: membrane_system
  real :: native_rotation(3,3), native_translation(3), native_metrics(6), native_matching(3)
  real :: native_ph, native_temperature, first_z
  logical :: prior_invalid, prior_zero, invalid_result, zero_result

  status=1
  ppm_status=0
  rotation=0.0_c_double
  translation=0.0_c_double
  metrics=0.0_c_double
  matching=0.0_c_double
  if(n<4 .or. n>150000 .or. path_length<1 .or. path_length>4096) return
  if(topology/=0 .and. topology/=1) return
  if(.not.ieee_is_finite(ph) .or. .not.ieee_is_finite(temperature)) return
  if(ph<0.0_c_double .or. ph>14.0_c_double .or. temperature<=0.0_c_double) return
  if(temperature>real(huge(1.0),c_double)) return
  do i=1,n
    if(.not.all(ieee_is_finite(input_xyz(:,i)))) return
    if(any(abs(input_xyz(:,i))>real(huge(1.0),c_double))) return
  end do
  allocate(xyz(3,n),atom_names(n),residue_names(n),chains(n),ids(n),stat=alloc_status)
  if(alloc_status/=0) then
    status=4
    return
  end if
  do i=1,n
    xyz(:,i)=real(input_xyz(:,i))
    ids(i)=residue_ids(i)
    chains(i)=chain_bytes(i)
    do j=1,4
      atom_names(i)(j:j)=atom_bytes(4*(i-1)+j)
      residue_names(i)(j:j)=residue_bytes(4*(i-1)+j)
    end do
  end do
  residue_library=' '
  do i=1,path_length
    residue_library(i:i)=path_bytes(i)
  end do
  do j=1,3
    membrane_system(j:j)=system_bytes(j)
  end do
  native_ph=real(ph)
  native_temperature=real(temperature)
  call ieee_get_flag(ieee_invalid,prior_invalid)
  call ieee_get_flag(ieee_divide_by_zero,prior_zero)
  call ieee_set_flag(ieee_invalid,.false.)
  call ieee_set_flag(ieee_divide_by_zero,.false.)
  call ppm3_driver(n,xyz,ids,chains,atom_names,residue_names, &
                   residue_library(:path_length),native_ph,native_temperature,membrane_system, &
                   native_rotation,native_translation,native_metrics,native_matching)
  call ieee_get_flag(ieee_invalid,invalid_result)
  call ieee_get_flag(ieee_divide_by_zero,zero_result)
  call ieee_set_flag(ieee_invalid,prior_invalid)
  call ieee_set_flag(ieee_divide_by_zero,prior_zero)
  status=ppm_status
  if(status/=0) return
  if(invalid_result.or.zero_result) then
    status=6
    return
  end if
  if(.not.all(ieee_is_finite(native_rotation)) .or. &
     .not.all(ieee_is_finite(native_translation)) .or. &
     .not.all(ieee_is_finite(native_metrics)) .or. &
     .not.all(ieee_is_finite(native_matching))) then
    status=6
    return
  end if
  ! The first scored atom is the canonical backbone N.
  ! write_pdb applies the same proper 180-degree X flip after projection.
  first_z=xyz(3,1)
  if((topology==0 .and. first_z>=0.0) .or. &
     (topology==1 .and. first_z<0.0)) then
    native_rotation(2:3,:)=-native_rotation(2:3,:)
    native_translation(2:3)=-native_translation(2:3)
  end if
  rotation=real(native_rotation,c_double)
  translation=real(native_translation,c_double)
  metrics=real(native_metrics,c_double)
  matching=real(native_matching,c_double)
end subroutine ppm3_orient
