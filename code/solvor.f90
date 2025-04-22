subroutine solvor
    
    use global_var
    implicit none
    
    integer :: i,j
    real*8 :: dvorx2,dvory2,dvorx1,dvory1,dtx1
    
    do i = 2,imax-1
        do j = 2,jmax-1
            dvorx2 = (vor(i+1,j)-2.d0*vor(i,j)+vor(i-1,j))/(dx**2.d0)
            dvory2 = (vor(i,j+1)-2.d0*vor(i,j)+vor(i,j-1))/(dy**2.d0)
            dtx1   = (T(i+1,j)-T(i-1,j))/(2.d0*dx)
            dvorx1 = u(i,j)*(vor(i+1,j)-vor(i-1,j))/(2.d0*dx)
            dvory1 = v(i,j)*(vor(i,j+1)-vor(i,j-1))/(2.d0*dy)
            rvor(i,j) = (dvorx2+dvory2)*Pr - Ra*Pr*dtx1 - dvorx1 - dvory1
        end do
    end do
    
    return
end subroutine solvor