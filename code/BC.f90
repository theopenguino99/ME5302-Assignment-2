subroutine BC

    use global_var
	implicit none

    integer :: i,j
    
    !do j = 2,jmax-1
    !    p(2,j)=0.25d0*p(3,j)
    !    p(imax-1,j)=0.25d0*p(imax-2,j)
    !end do
    !
    !do i = 2,imax-1
    !    p(i,2)=0.25d0*p(i,3)
    !    p(i,jmax-1)=0.25d0*p(i,jmax-2)
    !end do
    
    do j = 1,jmax
        vor(1,j)=2.d0*p(2,j)/(dx**2)
        vor(imax,j)=2.d0*p(imax-1,j)/(dx**2)
    end do

    do i = 2,imax-1
        vor(i,1)=2.d0*p(i,2)/(dy**2)
        vor(i,jmax)=2.d0*p(i,jmax-1)/(dy**2)
    end do
    

	return
end subroutine BC
    
