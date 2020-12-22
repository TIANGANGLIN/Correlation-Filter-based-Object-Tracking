"""
Python re-implementation of "Visual Object Tracking using Adaptive Correlation Filters"
@inproceedings{Bolme2010Visual,
  title={Visual object tracking using adaptive correlation filters},
  author={Bolme, David S. and Beveridge, J. Ross and Draper, Bruce A. and Lui, Yui Man},
  booktitle={Computer Vision & Pattern Recognition},
  year={2010},
}
"""
import numpy as np
import cv2,os

class mosse():
    def __init__(self,interp_factor=0.125,sigma=2.,img_path = 'datasets/surfer/'):
        # super(MOSSE).__init__()
        self.interp_factor=interp_factor
        self.sigma=sigma
        self.frame_list = _get_img_lists(img_path)
        self.frame_list.sort()
        self.first_frame = cv2.imread(self.frame_list[0])

    def init(self,bbox_init_gt=[228,118,140,174],chose_ROI=False,num_pretrain=128):
        if chose_ROI:
            bbox = cv2.selectROI('demo', self.first_frame, False, False)
        else:
            bbox = bbox_init_gt
        if len(self.first_frame.shape)!=2:
            assert self.first_frame.shape[2]==3
            self.first_frame=cv2.cvtColor(self.first_frame,cv2.COLOR_BGR2GRAY)
        self.first_frame=self.first_frame.astype(np.float32)/255
        x,y,w,h=tuple(bbox)
        self._center=(x+w/2,y+h/2)
        self.w,self.h=w,h
        w,h=int(round(w)),int(round(h))
        
        self._fi=cv2.getRectSubPix(self.first_frame,(w,h),self._center)
        self._G=np.fft.fft2(self._get_gauss_response((w,h),self.sigma))

        self._Ai=np.zeros_like(self._G)
        self._Bi=np.zeros_like(self._G)
        for _ in range(num_pretrain):
            fi=self._rand_warp(self._fi)
            Fi=np.fft.fft2(self._preprocessing(fi))
            self._Ai+=self._G*np.conj(Fi)
            self._Bi+=Fi*np.conj(Fi)

        # # pre train the filter on the first frame
        # Fi=np.fft.fft2(self._preprocessing(fi))
        # Ai = self._G * Fi
        # Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))

        # self._Ai = self.interp_factor*Ai
        # self._Bi = self.interp_factor*Bi

        # # self._Ai=np.zeros_like(self._G)
        # # self._Bi=np.zeros_like(self._G)
        # for _ in range(num_pretrain):
        #     fi=self._rand_warp(self._fi)
        #     Fi = np.fft.fft2(fi)
        #     self._Ai+=self._G*np.conj(Fi)
        #     self._Bi+=Fi*np.conj(Fi)

    def update(self,vis=False):
        for idx in range(len(self.frame_list)):
            current_frame = cv2.imread(self.frame_list[idx])
            current_frame_BGR=current_frame
            if len(current_frame.shape)!=2:
                assert current_frame.shape[2]==3
                current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
            current_frame=current_frame.astype(np.float32)/255
            Hi=self._Ai/self._Bi
            fi=cv2.getRectSubPix(current_frame,(int(round(self.w)),int(round(self.h))),self._center)
            fi=self._preprocessing(fi)
            Gi=Hi*np.fft.fft2(fi)
            gi=np.real(np.fft.ifft2(Gi))
            if vis is True:
                self.score=gi
            curr=np.unravel_index(np.argmax(gi, axis=None),gi.shape)
            dy,dx=curr[0]-(self.h/2),curr[1]-(self.w/2)
            x_c,y_c=self._center
            x_c+=dx
            y_c+=dy
            self._center=(x_c,y_c)
            fi=cv2.getRectSubPix(current_frame,(int(round(self.w)),int(round(self.h))),self._center)
            fi=self._preprocessing(fi)
            Fi=np.fft.fft2(fi)
            self._Ai=self.interp_factor*(self._G*np.conj(Fi))+(1-self.interp_factor)*self._Ai
            self._Bi=self.interp_factor*(Fi*np.conj(Fi))+(1-self.interp_factor)*self._Bi
            
            # visualize the tracking process...
            cv2.rectangle(current_frame_BGR, (int(self._center[0]-self.w/2),int(self._center[1]-self.h/2)), (int(self._center[0]+self.w/2),int(self._center[1]+self.h/2)), (255, 0, 0), 2)
            cv2.imshow('demo', current_frame_BGR)
            cv2.waitKey(1)
            # print([self._center[0]-self.w/2,self._center[1]-self.h/2,self.w,self.h])

    def _preprocessing(self,img,eps=1e-5):
        img=np.log(img+1)
        img=(img-np.mean(img))/(np.std(img)+eps)
        cos_window = self._window_func_2d(int(round(self.w)),int(round(self.h)))
        return cos_window*img

    # def _window_func_2d(self,height, width):
    def _window_func_2d(self,width,height):
        win_col = np.hanning(width)
        win_row = np.hanning(height)
        mask_col, mask_row = np.meshgrid(win_col, win_row)
        return mask_col * mask_row

    def _get_gauss_response(self,size,sigma):
        # self._G=np.fft.fft2(gaussian2d_labels((w,h),self.sigma))
        w,h=size

        # get the mesh grid
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))

        # get the center of the object
        center_x, center_y = w / 2, h / 2

        # cal the distance...
        dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (2*sigma**2)

        # get the response map
        response = np.exp(-dist)

        # normalize
        response = _linear_mapping(response)
        return response

    def _rand_warp(self,img):
        h, w = img.shape[:2]
        C = .1
        ang = np.random.uniform(-C, C)
        c, s = np.cos(ang), np.sin(ang)
        W = np.array([[c + np.random.uniform(-C, C), -s + np.random.uniform(-C, C), 0],
                      [s + np.random.uniform(-C, C), c + np.random.uniform(-C, C), 0]])
        center_warp = np.array([[w / 2], [h / 2]])
        tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
        W[:, 2:] = center_warp - center_warp * tmp
        warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
        return warped

##############################################
# utils 
##############################################
# it will extract the image list 
def _get_img_lists(img_path):
    frame_list = []
    for frame in os.listdir(img_path):
        if os.path.splitext(frame)[1] == '.jpg':
            frame_list.append(os.path.join(img_path, frame)) 
    return frame_list

def _linear_mapping(img):
        return (img - img.min()) / (img.max() - img.min())



if __name__ == "__main__":

    init_gt=[228,118,140,174]
    img_path = 'datasets/surfer/'

    tracker = mosse(img_path=img_path)
    tracker.init(bbox_init_gt=init_gt,chose_ROI=False)
    tracker.update()
