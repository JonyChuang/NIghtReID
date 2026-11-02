"""Original Convpass with a rectangular patch-grid interface only."""
import torch
from .convpass_official import Convpass as OfficialConvpass


class RectangularConvpass(OfficialConvpass):
    def forward(self,x,grid):
        batch,length,channels=x.shape
        height,width=grid
        if channels!=768 or length!=1+height*width:
            raise ValueError('Expected 768D CLS plus H*W patch tokens')
        x_down=self.act(self.adapter_down(x))
        patch=x_down[:,1:].reshape(batch,height,width,self.dim).permute(0,3,1,2)
        patch=self.adapter_conv(patch).permute(0,2,3,1).reshape(batch,height*width,self.dim)
        cls=x_down[:,:1].reshape(batch,1,1,self.dim).permute(0,3,1,2)
        cls=self.adapter_conv(cls).permute(0,2,3,1).reshape(batch,1,self.dim)
        x_down=self.act(torch.cat((cls,patch),dim=1))
        return self.adapter_up(self.dropout(x_down))
