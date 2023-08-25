# Internship

  This project focus on the implementation of a deep learning model to do the segmentation of vessels in mammograms in order to improve BAC detection.
To run the code you need to follow the stop of the github at "Codes\github.txt" to have the vesselness map from a mammogram. (The file "Codes\image_mammo" converts a dicom image to png). Then if the BAC segmentation is done you can run the file "Codes\follow_paths_2.py" to draw the extension of the BAC segmentation by maxmize the vesselness map. You can play with the patch size depending to your image to obatin better results. After that you can compare the likelihood percentage between the image obtained and the mask drawn by the expert using the file "Codes\pourcentage" that uses Jaccard's coefficient to compute the likelihood. And the file "Codes\nii.py" is used to convert .nii to .png as the masks are .nii.

The pdf file "recap" includes a more detailed summary with graphs showing the results.

The presentations of the meetings are here :

16/06 : https://tome.app/nimporte-812/presentation-1606-clixtn3wi0itwod3ccrica005

03/07 : https://tome.app/nimporte-812/presentation-0307-cljmjtp6s09hioa3anavzhfpe

17/07 : https://tome.app/nimporte-812/presentation-1707-cljmjtp6s09hioa3anavzhfpe

28/07 : https://tome.app/nimporte-812/presentation-2807-clkm9l0z500vqoe5rxakk4gwk

09/08 : https://tome.app/nimporte-812/presentation-0908-cll3erx2l0185mv5pkpkmuaxp

21/08 : https://tome.app/nimporte-812/presentation-2108-cllkjpk020ba6oc5pk551vtel
