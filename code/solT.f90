subroutine solT
    
    use global_var
    implicit none
    
    integer :: i,j
    real*8 :: dtx2,dty2,dtx1,dty1
    
    do i = 2,imax-1
        do j = 2,jmax-1
            dtx2 = (T(i+1,j)-2.d0*T(i,j)+T(i-1,j))/(dx**2.d0)
            dty2 = (T(i,j+1)-2.d0*T(i,j)+T(i,j-1))/(dy**2.d0)
            dtx1   = u(i,j)*(T(i+1,j)-T(i-1,j))/(2.d0*dx)
            dty1   = v(i,j)*(T(i,j+1)-T(i,j-1))/(2.d0*dy)

            rT(i,j) = dtx2 + dty2 - dtx1 - dty1
        end do
    end do
    
    return
end subroutine solT