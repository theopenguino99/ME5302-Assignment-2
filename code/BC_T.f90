subroutine BC_T

    use global_var
	implicit none

    integer :: i,j
    
    do j = 2,jmax-1
        T(1,j)=(4.d0*T(2,j)-T(3,j))/3.d0
        T(imax,j)=(4.d0*T(imax-1,j)-T(imax-2,j))/3.d0
    end do
    

	return
end subroutine BC_T
    
