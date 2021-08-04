import torch
import torch.nn.functional as F
from .nograd_module import NogradFunction
from ._dct import dct_2d, idct_2d


# https://en.wikipedia.org/wiki/JPEG
class JPGQuantizeFun(NogradFunction):
    def __init__(self, jpg_quality=0.3):
        super(JPGQuantizeFun, self).__init__()
        self.jpg_quality = jpg_quality
        self.patchsize = 8
        self.QmatY, self.QmatCrCb = self.getQuantizeMat(jpg_quality=jpg_quality)
        self.lvl = 255

    def jpg_compress(self, img):
        # img is [0,1]

        imsize = img.shape

        # rgb to ycbcr
        img255 = img * self.lvl
        if imsize[1] == 3:
            img255 = self.rgb2ycbcr(img255)
            img255 = torch.round(img255)  # necessary?

        # image to 8x8 blocks
        img_unf = F.unfold(img255, kernel_size=(self.patchsize, self.patchsize),
                           stride=(self.patchsize, self.patchsize))
        unf_shape = img_unf.shape
        img_unf = img_unf.view([unf_shape[0], imsize[1], self.patchsize, self.patchsize, unf_shape[2]])

        # dct
        img_unf = img_unf.transpose(2, 4).contiguous()
        img_unf = dct_2d(img_unf, 'ortho')
        img_unf = torch.round(img_unf)

        # quantization with matrix
        Qmat_list = [self.QmatY, self.QmatCrCb, self.QmatCrCb]
        for ii in range(img_unf.shape[1]):
            img_unf[:, ii, :, :, :] = self.quantizeBlocks(img_unf[:, ii, :, :, :], Qmat_list[ii])
        img_unf = torch.round(img_unf)

        # inverse quantization with matrix
        for ii in range(img_unf.shape[1]):
            img_unf[:, ii, :, :, :] = self.quantizeBlocks(img_unf[:, ii, :, :, :], 1 / Qmat_list[ii])

        # idct
        img_unf = idct_2d(img_unf, 'ortho')
        img_unf = torch.round(img_unf)
        img_unf = img_unf.transpose(2, 4).contiguous()

        # 8x8 blocks to iamge
        img_unf = img_unf.view(unf_shape)
        img_out = F.fold(img_unf, output_size=imsize[2:], kernel_size=(self.patchsize, self.patchsize),
                         stride=(self.patchsize, self.patchsize))

        if imsize[1] == 3:
            img_out = self.ycbcr2rgb(img_out)
            img_out = torch.round(img_out)
        img_out /= self.lvl

        return img_out

    def quantizeBlocks(self, blocks, Qmat):
        blocks_shape = blocks.shape

        blocks = blocks.contiguous().view(-1, blocks_shape[-2], blocks_shape[-1])
        for ii in range(blocks.shape[-2]):
            for jj in range(blocks_shape[-1]):
                blocks[:, ii, jj] /= Qmat[ii, jj]

        blocks = blocks.view(blocks_shape)

        return blocks

    def rgb2ycbcr(self, im):
        # https://en.wikipedia.org/wiki/YCbCr
        # https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
        r = im[:, 0, :, :]
        g = im[:, 1, :, :]
        b = im[:, 2, :, :]

        # Y
        Y = .299 * r + .587 * g + .114 * b
        # Cb
        Cb = 128 - .168736 * r - .331264 * g + .5 * b
        # Cr
        Cr = 128 + .5 * r - .418688 * g - .081312 * b

        if len(Y.shape) == 3:
            Y = Y.view([Y.shape[0], -1, Y.shape[-2], Y.shape[-1]])
            Cb = Cb.view([Cb.shape[0], -1, Cb.shape[-2], Cb.shape[-1]])
            Cr = Cr.view([Cr.shape[0], -1, Cr.shape[-2], Cr.shape[-1]])

        im_out = torch.cat([Y, Cb, Cr], 1)

        return im_out

    def ycbcr2rgb(self, im):
        # https://en.wikipedia.org/wiki/YCbCr
        # https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb

        y = im[:, 0, :, :]
        cb = im[:, 1, :, :] - 128
        cr = im[:, 2, :, :] - 128

        # R
        r = y + 1.402 * cr
        # G
        g = y - .344136 * cb - .714136 * cr
        # B
        b = y + 1.772 * cb

        if len(r.shape) == 3:
            r = r.view([r.shape[0], -1, r.shape[-2], r.shape[-1]])
            g = g.view([g.shape[0], -1, g.shape[-2], g.shape[-1]])
            b = b.view([b.shape[0], -1, b.shape[-2], b.shape[-1]])

        im_out = torch.cat([r, g, b], 1)
        im_out = torch.clamp(im_out, min=0, max=255)

        return im_out

    def quantize(self, input):
        output = torch.clamp(input, min=0, max=1)
        output = torch.mul(output, self.lvl)
        output = torch.round(output)
        output = torch.mul(output, 1 / self.lvl)

        return output

    # get quantization matrix for Y and CrCb
    def getQuantizeMat(self, jpg_quality):
        mat50Y = torch.Tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                               [12, 12, 14, 19, 26, 58, 60, 55],
                               [14, 13, 16, 24, 40, 57, 69, 56],
                               [14, 17, 22, 29, 51, 87, 80, 62],
                               [18, 22, 37, 56, 68, 109, 103, 77],
                               [24, 35, 55, 64, 81, 104, 113, 92],
                               [49, 64, 78, 87, 103, 121, 120, 101],
                               [72, 92, 95, 98, 112, 100, 103, 99]])
        mat50CrCb = torch.Tensor([[17, 18, 24, 47, 99, 99, 99, 99],
                                  [18, 21, 26, 66, 99, 99, 99, 99],
                                  [24, 26, 56, 99, 99, 99, 99, 99],
                                  [47, 66, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99],
                                  [99, 99, 99, 99, 99, 99, 99, 99]])

        # mat100 = torch.ones([8, 8])

        if jpg_quality < 0.01: jpg_quality = 0.01
        if jpg_quality > 1: jpg_quality = 1
        if jpg_quality < 0.5:
            scale_factor = 5000 / (jpg_quality * 100)
        else:
            scale_factor = 200 - (jpg_quality * 100) * 2

        matY = (mat50Y * scale_factor + 50) / 100
        matY = torch.clamp(torch.floor(matY), min=1, max=255)

        matCrCb = (mat50CrCb * scale_factor + 50) / 100
        matCrCb = torch.clamp(torch.floor(matCrCb), min=1, max=255)

        return matY, matCrCb

    def quantize(self, input):
        # input and output is [0, 1]

        output = torch.clamp(input, min=0, max=1)
        output = torch.mul(output, self.lvl)
        output = torch.round(output)
        output = torch.mul(output, 1 / self.lvl)

        return output

    def forward(self, input):
        if self.QmatY.data.type() != input.data.type():
            if input.is_cuda:
                self.QmatY = self.QmatY.cuda(input.get_device())
                self.QmatCrCb = self.QmatCrCb.cuda(input.get_device())
            self.QmatY = self.QmatY.type_as(input)
            self.QmatCrCb = self.QmatCrCb.type_as(input)

        output = self.jpg_compress(input)
        return output


def jpeg(x, q):
    fn = JPGQuantizeFun(jpg_quality=q)
    return fn(x)
