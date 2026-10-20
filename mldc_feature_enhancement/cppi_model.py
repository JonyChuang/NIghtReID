"""Global MLDC/CNL model and historical CNN/CPPI checkpoint models."""
import torch
from torch import nn
from .model import FeatEnHancerMLDC,add_config as add_base_config,objectives
from .cppi_modules import CNNPyramid,PatchPyramid,CTILite

VARIANT='cppi_lite_mrfp_learnable_budget_20261018'


def add_config(cfg):
    cfg=add_base_config(cfg)
    cfg.MODEL.CPPI_ENABLED=True
    cfg.MODEL.CPPI_SOURCE='cnn'
    cfg.MODEL.CPPI_LAYERS=[2,4,6,8]
    cfg.MODEL.CPPI_DIM=192
    cfg.MODEL.CPPI_HEADS=4
    cfg.MODEL.CPPI_INITIAL_RATIO=.05
    cfg.MODEL.CPPI_UP_INIT_STD=.01
    cfg.SOLVER.CPPI_LR_FACTOR=10.
    cfg.TEST.CPPI_VISUALIZE=True
    cfg.TEST.CPPI_VIS_INDICES=[810,1544,11,2059,160,24]
    cfg.MODEL.CNN_SD_ENABLED=False
    cfg.MODEL.CNN_SD_CHECKPOINT_READER=True
    cfg.MODEL.CNN_SD_QUERY='fused_cls'
    cfg.SOLVER.CNN_SD_LR_FACTOR=10.
    cfg.SOLVER.CNN_SD_READER_LR_FACTOR=1.
    cfg.SOLVER.SHALLOW_LOSS_WEIGHT=0.
    cfg.SOLVER.CNL_WEIGHT=.1
    cfg.SOLVER.CNL_SHALLOW_WEIGHT=.5
    cfg.SOLVER.CNL_GAMMA=.9
    cfg.SOLVER.CNL_TEMPERATURE=.05
    cfg.SOLVER.CNL_MOMENTUM=.2
    return cfg


def config_from_dump(base_cfg,text):
    cfg=add_config(base_cfg.clone());stored=type(cfg).load_cfg(text)
    if stored.MODEL.get('ENHANCEMENT_MODE')=='cnn_shallow_deep' and 'CNN_SD_QUERY' not in stored.MODEL:
        # Old checkpoints used CLS11. Never reinterpret their learned readout.
        cfg.MODEL.CNN_SD_QUERY='cls11'
    stored.MODEL.DEVICE_ID=repr(str(stored.MODEL.DEVICE_ID))
    cfg.merge_from_other_cfg(stored)
    return cfg


class CPPIReID(FeatEnHancerMLDC):
    def __init__(self,cfg,num_classes,cameras,views,initialize=True):
        plain=cfg.clone();plain.defrost()
        plain.MODEL.ENHANCEMENT_MODE='patch_feature';plain.MODEL.FEATENHANCER_ENABLED=False;plain.freeze()
        super().__init__(plain,num_classes,cameras,views,initialize=initialize)
        if cfg.MODEL.ENHANCEMENT_MODE!='cppi_lite' or cfg.MODEL.FEATENHANCER_ENABLED:
            raise ValueError('CPPI uses the plain SSL backbone without A/B/PFE branches')
        self.mode='cppi_lite';self.enabled=bool(cfg.MODEL.CPPI_ENABLED)
        self.cppi_source=cfg.MODEL.CPPI_SOURCE;self.cppi_layers=tuple(cfg.MODEL.CPPI_LAYERS)
        if self.cppi_source not in ('cnn','vit_pyramid'):
            raise ValueError('CPPI source is cnn or vit_pyramid')
        if not self.cppi_layers or list(self.cppi_layers)!=sorted(set(self.cppi_layers)) or not all(1<=n<9 for n in self.cppi_layers):
            raise ValueError('Use sorted unique insertion depths before CLS9-12')
        self.grid=(self.base.patch_embed.num_y,self.base.patch_embed.num_x)
        self.cppi_settings=dict(dim=int(cfg.MODEL.CPPI_DIM),heads=int(cfg.MODEL.CPPI_HEADS),
            initial_ratio=float(cfg.MODEL.CPPI_INITIAL_RATIO),up_init_std=float(cfg.MODEL.CPPI_UP_INIT_STD))
        self.cppi_injectors=nn.ModuleDict();self.cppi_pyramid=nn.Identity()
        self._cppi_features=None;self._cppi_stats={};self._cls_attention={};self._diagnostics=False;self._handles=[]
        if self.enabled:
            dim=self.cppi_settings['dim']
            self.cppi_pyramid=CNNPyramid(dim) if self.cppi_source=='cnn' else PatchPyramid(self.base.in_planes,dim)
            for depth in self.cppi_layers:
                self.cppi_injectors[str(depth)]=CTILite(self.base.in_planes,**self.cppi_settings)
                self._handles.append(self.base.blocks[depth-1].register_forward_hook(self._stage_hook(depth)))
        for depth in (9,10,11,12):
            self._handles.append(self.base.blocks[depth-1].attn.attn_drop.register_forward_hook(self._attention_hook(depth)))

    def _stage_hook(self,depth):
        def hook(module,args,output):
            if not self.enabled:return output
            patches=output[:,1:]
            sources=self._cppi_features if self.cppi_source=='cnn' else self.cppi_pyramid(patches,self.grid)
            enhanced,stats=self.cppi_injectors[str(depth)](patches,sources,self.grid,self._diagnostics)
            if self._diagnostics:self._cppi_stats[depth]=stats
            return torch.cat((output[:,:1],enhanced),1)
        return hook

    def _attention_hook(self,depth):
        def hook(module,args,output):
            if self._diagnostics:
                self._cls_attention[depth]=output[:,:,0,1:].detach().float().mean(1).reshape(-1,*self.grid)
        return hook

    def forward(self,images,label=None,cam_label=None,view_label=None,diagnostics=False):
        self._diagnostics=bool(diagnostics);self._cppi_stats={};self._cls_attention={}
        try:
            if self.enabled and self.cppi_source=='cnn':self._cppi_features=self.cppi_pyramid(images)
            layers=self.base(images,cam_label=cam_label,view_label=view_label)
            raw=self.mldc(layers);neck=self.bottleneck(raw)
            if self.training or diagnostics:
                data=dict(retrieval=neck,raw_readout=raw,global_feature=raw,layers=layers,enhanced=images)
                if self.training:data['scores']=self.classifier(self.dropout(neck))
                if diagnostics:
                    data['cls_attention']=torch.stack([self._cls_attention[d] for d in (9,10,11,12)],1)
                    if self.enabled:
                        for key in ('before_norm','after_norm','correction_norm','correction_ratio','budget','scale_mass','local_entropy'):
                            data['cppi_'+key]=torch.stack([self._cppi_stats[d][key] for d in self.cppi_layers],1)
                return data
            return neck
        finally:
            self._cppi_features=None;self._cppi_stats={};self._cls_attention={};self._diagnostics=False

    def correction_budgets(self):
        return {depth:float(module.correction_budget().detach().cpu()) for depth,module in self.cppi_injectors.items()}

    def architecture(self):
        return dict(variant=VARIANT,enhancer='CoMer-inspired one-way CNN/MRFP + CTI-Lite; not original deformable CTI',
            cppi_enabled=self.enabled,source=self.cppi_source,cppi_layers=list(self.cppi_layers),**self.cppi_settings,
            source_scales=[8,16,32],local_candidates=[9,9,9],cnn_computed_once=self.cppi_source=='cnn',
            normalization='CNN/MRFP GroupNorm; CTI LayerNorm; FP32 magnitude',
            budget_mode='one learned positive softplus scalar per stage; initial_ratio is initialization, no upper clamp',
            grid=list(self.grid),global_layers=[9,10,11,12],global_weights=[.25]*4,global_fusion='fixed_equal',
            descriptor_dim=768,spatial_partition=False,cls_directly_modified=False,joint_from_epoch=1,
            retrieval='BN-after global; evaluator L2')


CNN_SDCL_VARIANT='cnn_sdcl_supervised_cnl_20261019'
CNN_FUSED_VARIANT='cnn_fused_cls_guided_no_cnl_20261019'
GLOBAL_CNL_VARIANT='mldc_global_cnl_20261020'


class MLDCCNLReID(FeatEnHancerMLDC):
    """One fused CLS descriptor. CNL is a training-only loss, not a model branch."""
    def __init__(self,cfg,num_classes,cameras,views,initialize=True):
        if cfg.MODEL.ENHANCEMENT_MODE!='mldc_cnl' or cfg.MODEL.FEATENHANCER_ENABLED or cfg.MODEL.CNN_SD_ENABLED:
            raise ValueError('Use mldc_cnl with both image enhancement and CNN shallow branch disabled')
        plain=cfg.clone();plain.defrost()
        plain.MODEL.ENHANCEMENT_MODE='patch_feature';plain.MODEL.FEATENHANCER_ENABLED=False;plain.freeze()
        super().__init__(plain,num_classes,cameras,views,initialize=initialize)
        self.mode='mldc_cnl';self.enabled=False

    def forward(self,images,label=None,cam_label=None,view_label=None,diagnostics=False):
        from torch.nn import functional as F
        layers=self.base(images,cam_label=cam_label,view_label=view_label)
        raw=self.mldc(layers);neck=self.bottleneck(raw)
        unit=F.normalize(neck.float(),dim=1)
        if self.training or diagnostics:
            data=dict(global_raw=raw,global_neck=neck,global_unit=unit,retrieval=unit,
                raw_readout=raw,global_feature=raw,layers=layers)
            if self.training:data['scores']=self.classifier(self.dropout(neck))
            return data
        return unit

    def architecture(self):
        return dict(variant=GLOBAL_CNL_VARIANT,cnn_enabled=False,shallow_enabled=False,
            global_layers=[9,10,11,12],global_fusion='fixed_equal',global_weights=[.25]*4,
            descriptor_dim=self.base.in_planes,patch_injection=False,spatial_partition=False,
            joint_from_epoch=1,retrieval='single BN-after fused CLS; L2 normalization')


class CNNShallowDeepReID(FeatEnHancerMLDC):
    """Independent CNN readout; no CTI hooks or patch injection.

    CPPIReID above remains available for historical checkpoint evaluation.
    """
    def __init__(self,cfg,num_classes,cameras,views,initialize=True):
        import copy
        from .cppi_modules import CNNTokenReader
        if cfg.MODEL.ENHANCEMENT_MODE!='cnn_shallow_deep' or cfg.MODEL.FEATENHANCER_ENABLED:
            raise ValueError('Use cnn_shallow_deep with the plain SSL input')
        plain=cfg.clone();plain.defrost()
        plain.MODEL.ENHANCEMENT_MODE='patch_feature'
        plain.MODEL.FEATENHANCER_ENABLED=False;plain.freeze()
        super().__init__(plain,num_classes,cameras,views,initialize=initialize)
        self.mode='cnn_shallow_deep';self.enabled=bool(cfg.MODEL.CNN_SD_ENABLED)
        self.query_source=cfg.MODEL.CNN_SD_QUERY
        if self.query_source not in ('fused_cls','cls11'):
            raise ValueError('CNN_SD_QUERY must be fused_cls or the historical cls11 control')
        self.checkpoint_reader=bool(cfg.MODEL.CNN_SD_CHECKPOINT_READER)
        if self.enabled:
            self.cppi_pyramid=CNNPyramid(int(cfg.MODEL.CPPI_DIM))
            self.cnn_reader=CNNTokenReader(int(cfg.MODEL.CPPI_DIM),self.base.in_planes,
                copy.deepcopy(self.base.blocks[-1]),copy.deepcopy(self.base.norm))
            # The original backbone's final LN is unused/frozen; its copy is active.
            self.cnn_reader.requires_grad_(True)
            self.shallow_bottleneck=nn.BatchNorm1d(self.base.in_planes)
            self.shallow_bottleneck.bias.requires_grad_(False)
            self.shallow_classifier=nn.Linear(self.base.in_planes,num_classes,bias=False)
            nn.init.normal_(self.shallow_classifier.weight,std=.001)

    def forward(self,images,label=None,cam_label=None,view_label=None,diagnostics=False):
        import math
        from torch.nn import functional as F
        from torch.utils.checkpoint import checkpoint
        layers=self.base(images,cam_label=cam_label,view_label=view_label)
        global_raw=self.mldc(layers)
        global_neck=self.bottleneck(global_raw)
        g=F.normalize(global_neck.float(),dim=1)
        data=dict(global_raw=global_raw,global_neck=global_neck,global_unit=g,
                  raw_readout=global_raw,global_feature=global_raw,layers=layers)
        if self.enabled:
            pyramid=self.cppi_pyramid(images)
            tokens=self.cnn_reader.make_tokens(pyramid)
            # Feed the fused deep CLS into the CNN readout before BN/L2.
            # Keep the path differentiable so shallow supervision reaches CLS9-12.
            query=global_raw if self.query_source=='fused_cls' else layers[10]
            if self.training and self.checkpoint_reader:
                shallow_raw=checkpoint(self.cnn_reader.read,query,tokens,use_reentrant=False)
            else:
                shallow_raw=self.cnn_reader.read(query,tokens)
            shallow_neck=self.shallow_bottleneck(shallow_raw)
            s=F.normalize(shallow_neck.float(),dim=1)
            data.update(shallow_raw=shallow_raw,shallow_neck=shallow_neck,shallow_unit=s,
                        retrieval=torch.cat((g,s),dim=1)/math.sqrt(2.))
            if diagnostics:
                data['cnn_token_grids']=[list(feature.shape[-2:]) for feature in pyramid]
        else:
            data['retrieval']=g
        if self.training:
            data['scores']=self.classifier(self.dropout(global_neck))
            if self.enabled:
                data['shallow_scores']=self.shallow_classifier(self.dropout(shallow_neck))
        return data if self.training or diagnostics else data['retrieval']

    def architecture(self):
        fused=self.query_source=='fused_cls'
        return dict(variant=CNN_FUSED_VARIANT if fused else CNN_SDCL_VARIANT,
            cnn_enabled=self.enabled,source='independent CNN/MRFP',
            source_scales=[8,16,32],cnn_dim=self.cppi_pyramid.stem[-1][0].out_channels if self.enabled else None,
            query='raw fused CLS9-12 (fixed equal)' if fused else 'raw CLS11',
            reader='independent SSL-initialized final block and LN',
            position='normalized 2D sine/cosine plus learned scale embedding',
            global_layers=[9,10,11,12],global_fusion='fixed_equal',global_weights=[.25]*4,
            patch_injection=False,spatial_partition=False,joint_from_epoch=1,
            descriptor_dim=1536 if self.enabled else 768,
            retrieval='separate BN, per-branch L2, equal-energy concatenation')
