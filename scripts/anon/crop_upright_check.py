import sys, cv2, os, numpy as np
sys.path.insert(0,"/springbrook/share/eng/esrpxk/CogntiveGaze")
from video_preprocess.detector import VideoFaceDetector
b="/springbrook/share/eng/esrpxk/datasets"
print(f"{'rec':7}{'meta':>6}{'root':22}{'det':>7}{'tilt':>7}{'eyeY frac':>11}")
for R,meta in (("00012","0"),("00019","0"),("00020","180"),("00006","90"),("00010","90")):
    for root in ("ProcessedData","calib_support_K72"):
        d=f"{b}/{root}/{R}/appleFace"
        if not os.path.isdir(d): continue
        names=sorted(os.listdir(d)); names=names[::max(1,len(names)//5)][:5]
        det=VideoFaceDetector(refine_landmarks=True)
        hits,tilts,fr=0,[],[]
        for n in names:
            im=cv2.imread(f"{d}/{n}")
            lm=det.detect(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
            if lm is None: continue
            hits+=1
            le=np.asarray(lm.left_eye).mean(0); re=np.asarray(lm.right_eye).mean(0)
            ov=np.asarray(lm.face_oval)
            tilts.append(np.degrees(np.arctan2(re[1]-le[1],re[0]-le[0])))
            y0,y1=ov[:,1].min(),ov[:,1].max()
            fr.append(((le[1]+re[1])/2-y0)/max(1e-6,(y1-y0)))
        t=f"{np.median(tilts):+.0f}d" if tilts else "--"
        f_=f"{np.median(fr):.2f}" if fr else "--"
        print(f"{R:7}{meta:>6}{root:22}{hits}/{len(names):<5}{t:>7}{f_:>11}")
