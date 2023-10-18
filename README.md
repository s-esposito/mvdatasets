# DataSets

Various data loaders for multi-view datasets commonly used in 3D reconstruction and view-synthesis:
- [DTU]
- [NeRF-Synthetic]

Soon to be supported:
- [NeRF-LLFF]
- [NeRF-360]

## Misc:
    This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), the X-axis points right, and the Z-axis points into the image plane.
    
## Author
Stefano Esposito



#### Notes

- Ho una Dataset object contenente una lista di Camera object
- Camera object ha un metodo get_random_rays(N) che ritorna N rays a random passanti per i pixel della rispettiva immagine
    -- questo posso usarlo anche per prendere un solo raggio, ma nella stessa epoca non voglio samplare lo stesso due volte
Alla fine voglio un dataloader che mi crei dei batch di rays da camere diverse, ritornando: rays_o, rays_d, gt_rgb, cam_id  