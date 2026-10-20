import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def norm_map(x):
    x = x.float()
    x = x - x.min()
    return x / (x.max() + 1e-6)


def resize_map(x, image_h, image_w):
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(image_h, image_w), mode='bilinear', align_corners=False)
    return x.squeeze().cpu().numpy()


def token_to_heatmap(tokens, grid_h, grid_w, image_h, image_w):
    tokens = tokens.reshape(grid_h, grid_w)
    tokens = norm_map(tokens)
    return resize_map(tokens, image_h, image_w)

'''def visualize_qnorm_sample(
    image,
    teacher_qnorm,
    student_before,
    prior_gain,
    student_after,
    cls_attn,
    grid_h,
    grid_w,
    save_path
):
    image_h, image_w = image.shape[:2]

    maps = [
        teacher_qnorm,
        student_before,
        prior_gain,
        student_after,
        cls_attn
    ]

    maps = [
        token_to_heatmap(x, grid_h, grid_w, image_h, image_w)
        for x in maps
    ]

    titles = [
        'Original',
        'Teacher Q-Norm',
        'Student Q-Norm Before',
        'Teacher Prior Gain',
        'Student Q-Norm After',
        'Block12 CLS Attention'
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    for i, heatmap in enumerate(maps, start=1):
        axes[i].imshow(image)
        axes[i].imshow(heatmap, alpha=0.55, cmap='jet')
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()'''


def token_to_raw_map(tokens, grid_h, grid_w, image_h, image_w):
    tokens = tokens.float().reshape(1, 1, grid_h, grid_w)

    tokens = F.interpolate(
        tokens,
        size=(image_h, image_w),
        mode='bilinear',
        align_corners=False
    )

    return tokens.squeeze().cpu().numpy()


def normalize_map(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-6)

'''def visualize_qnorm_sample(
    image,
    teacher_qnorm,
    student_before,
    student_after,
    cls_attn_without,
    cls_attn_with,
    cls_attn_diff,
    grid_h,
    grid_w,
    save_path
):
    image_h, image_w = image.shape[:2]

    teacher_map = token_to_raw_map(
        teacher_qnorm, grid_h, grid_w, image_h, image_w
    )

    q_before = token_to_raw_map(
        student_before, grid_h, grid_w, image_h, image_w
    )

    q_after = token_to_raw_map(
        student_after, grid_h, grid_w, image_h, image_w
    )

    attn_without = token_to_raw_map(
        cls_attn_without, grid_h, grid_w, image_h, image_w
    )

    attn_with = token_to_raw_map(
        cls_attn_with, grid_h, grid_w, image_h, image_w
    )

    attn_diff = token_to_raw_map(
        cls_attn_diff, grid_h, grid_w, image_h, image_w
    )

    # ---------------------------------------------------------
    # 普通热力图
    # ---------------------------------------------------------
    teacher_map = normalize_map(teacher_map)

    # Student Before / After 使用相同颜色范围
    q_min = min(q_before.min(), q_after.min())
    q_max = max(q_before.max(), q_after.max())

    q_before = (q_before - q_min) / (q_max - q_min + 1e-6)
    q_after = (q_after - q_min) / (q_max - q_min + 1e-6)

    # CLS Attention Without / With 使用相同颜色范围
    attn_min = min(attn_without.min(), attn_with.min())
    attn_max = max(attn_without.max(), attn_with.max())

    attn_without = (
        attn_without - attn_min
    ) / (
        attn_max - attn_min + 1e-6
    )

    attn_with = (
        attn_with - attn_min
    ) / (
        attn_max - attn_min + 1e-6
    )

    # Difference 必须以 0 为中心
    diff_max = np.max(np.abs(attn_diff)) + 1e-6

    fig, axes = plt.subplots(1, 7, figsize=(21, 6))

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Teacher Q-Norm
    axes[1].imshow(image)
    axes[1].imshow(teacher_map, alpha=0.55, cmap='jet')
    axes[1].set_title('Teacher Q-Norm')
    axes[1].axis('off')

    # Student Q-Norm Before
    axes[2].imshow(image)
    axes[2].imshow(q_before, alpha=0.55, cmap='jet')
    axes[2].set_title('Student Q-Norm Before')
    axes[2].axis('off')

    # Student Q-Norm After
    axes[3].imshow(image)
    axes[3].imshow(q_after, alpha=0.55, cmap='jet')
    axes[3].set_title('Student Q-Norm After')
    axes[3].axis('off')

    # CLS Attention without Q-Norm
    axes[4].imshow(image)
    axes[4].imshow(attn_without, alpha=0.55, cmap='jet')
    axes[4].set_title('CLS Attention without Q-Norm')
    axes[4].axis('off')

    # CLS Attention with Q-Norm
    axes[5].imshow(image)
    axes[5].imshow(attn_with, alpha=0.55, cmap='jet')
    axes[5].set_title('CLS Attention with Q-Norm')
    axes[5].axis('off')

    # CLS Attention Difference
    axes[6].imshow(image)

    diff_img = axes[6].imshow(
        attn_diff,
        alpha=0.65,
        cmap='coolwarm',
        vmin=-diff_max,
        vmax=diff_max
    )

    axes[6].set_title('CLS Attention Difference')
    axes[6].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()'''

def visualize_qnorm_sample(image, teacher_qnorm, student_qnorm, qnorm_deficit,
                           cls_attn_without, cls_attn_with, cls_attn_diff,
                           grid_h, grid_w, save_path):

    image_h, image_w = image.shape[:2]

    teacher_map = token_to_raw_map(teacher_qnorm, grid_h, grid_w, image_h, image_w)
    student_map = token_to_raw_map(student_qnorm, grid_h, grid_w, image_h, image_w)
    deficit_map = token_to_raw_map(qnorm_deficit, grid_h, grid_w, image_h, image_w)
    attn_without = token_to_raw_map(cls_attn_without, grid_h, grid_w, image_h, image_w)
    attn_with = token_to_raw_map(cls_attn_with, grid_h, grid_w, image_h, image_w)
    attn_diff = token_to_raw_map(cls_attn_diff, grid_h, grid_w, image_h, image_w)

    teacher_map = normalize_map(teacher_map)
    student_map = normalize_map(student_map)
    deficit_map = normalize_map(deficit_map)

    # CLS Attention without / with guidance 使用相同的颜色范围
    attn_min = min(attn_without.min(), attn_with.min())
    attn_max = max(attn_without.max(), attn_with.max())
    attn_without = (attn_without - attn_min) / (attn_max - attn_min + 1e-6)
    attn_with = (attn_with - attn_min) / (attn_max - attn_min + 1e-6)

    # Difference 使用以 0 为中心的颜色范围
    diff_max = np.max(np.abs(attn_diff)) + 1e-6

    fig, axes = plt.subplots(1, 7, figsize=(21, 6))

    axes[0].imshow(image)
    axes[0].set_title('Original')

    axes[1].imshow(image)
    axes[1].imshow(teacher_map, alpha=0.55, cmap='jet')
    axes[1].set_title('Teacher Q-Norm')

    axes[2].imshow(image)
    axes[2].imshow(student_map, alpha=0.55, cmap='jet')
    axes[2].set_title('Student Q-Norm')

    axes[3].imshow(image)
    axes[3].imshow(deficit_map, alpha=0.55, cmap='jet')
    axes[3].set_title('Q-Norm Deficit')

    axes[4].imshow(image)
    axes[4].imshow(attn_without, alpha=0.55, cmap='jet')
    axes[4].set_title('CLS Attention without Guidance')

    axes[5].imshow(image)
    axes[5].imshow(attn_with, alpha=0.55, cmap='jet')
    axes[5].set_title('CLS Attention with Guidance')

    axes[6].imshow(image)
    axes[6].imshow(attn_diff, alpha=0.65, cmap='coolwarm', vmin=-diff_max, vmax=diff_max)
    axes[6].set_title('CLS Attention Difference')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()