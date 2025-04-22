subroutine caluv

    use global_var
	implicit none

    integer :: i,j
    
    do j = 1,jmax
        u(1,j)=0.d0
        v(1,j)=0.d0
        u(imax,j)=0.d0
        v(imax,j)=0.d0
    end do
    
    do i = 2,imax-1
        u(i,1)=0.d0
        v(i,1)=0.d0
        !T(i,1)=1.d0+0.5d0*cos(PI*x(i))
        !T(i,jmax)=0.d0
        u(i,jmax)=0.d0
        v(i,jmax)=0.d0
    end do

    do i = 2,imax-1
        do j = 2,jmax-1
            u(i,j)=0.5d0*(p(i,j+1)-p(i,j-1))/dy
            v(i,j)=0.5d0*(p(i-1,j)-p(i+1,j))/dx
        end do
    end do
    
	return
end subroutine caluv
    
