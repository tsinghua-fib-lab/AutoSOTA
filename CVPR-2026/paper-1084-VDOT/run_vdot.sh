
#------------------------ CLI ------------------------#
# txt2vid txt2vid
python inference/vace_wan_inference.py --prompt "狂风巨浪的大海，镜头缓缓推进，一艘渺小的帆船在汹涌的波涛中挣扎漂荡。海面上白沫翻滚，帆船时隐时现，仿佛随时可能被巨浪吞噬。天空乌云密布，雷声轰鸣，海鸥在空中盘旋尖叫。帆船上的人们紧紧抓住缆绳，努力保持平衡。画面风格写实，充满紧张和动感。近景特写，强调风浪的冲击力和帆船的摇晃"

# extension firstframe
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/firstframe/src_video.mp4" --src_mask "benchmarks/VACE-Benchmark/assets/examples/firstframe/src_mask.mp4" --prompt "纪实摄影风格，前景是一位中国越野爱好者坐在越野车上，手持车载电台正在进行通联。他五官清晰，表情专注，眼神坚定地望向前方。越野车停在户外，车身略显脏污，显示出经历过的艰难路况。镜头从车外缓缓拉近，最后定格在人物的面部特写上，展现出他的，动态镜头运镜。"

# repainting inpainting
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/inpainting/src_video.mp4" --src_mask "benchmarks/VACE-Benchmark/assets/examples/inpainting/src_mask.mp4" --prompt "一只巨大的金色凤凰从繁华的城市上空展翅飞过，羽毛如火焰般璀璨，闪烁着温暖的光辉，翅膀雄伟地展开。凤凰高昂着头，目光炯炯，轻轻扇动翅膀，散发出淡淡的光芒。下方是熙熙攘攘的市中心，人群惊叹，车水马龙，红蓝两色的霓虹灯在夜空下闪烁。镜头俯视城市街道，捕捉这一壮丽的景象，营造出既神秘又辉煌的氛围。"

# repainting outpainting
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_video.mp4" --src_mask "benchmarks/VACE-Benchmark/assets/examples/outpainting/src_mask.mp4" --prompt "赛博朋克风格，无人机俯瞰视角下的现代西安城墙，镜头穿过永宁门时泛起金色涟漪，城墙砖块化作数据流重组为唐代长安城。周围的街道上流动的人群和飞驰的机械交通工具交织在一起，现代与古代的交融，城墙上的灯光闪烁，形成时空隧道的效果。全息投影技术展现历史变迁，粒子重组特效细腻逼真。大远景逐渐过渡到特写，聚焦于城门特效。"

# control depth
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/depth/src_video.mp4" --prompt "一群年轻人在天空之城拍摄集体照。画面中，一对年轻情侣手牵手，轻声细语，相视而笑，周围是飞翔的彩色热气球和闪烁的星星，营造出浪漫的氛围。天空中，暖阳透过飘浮的云朵，洒下斑驳的光影。镜头以近景特写开始，随着情侣间的亲密互动，缓缓拉远。"

# control flow
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/flow/src_video.mp4" --prompt "纪实摄影风格，一颗鲜红的小番茄缓缓落入盛着牛奶的玻璃杯中，溅起晶莹的水花。画面以慢镜头捕捉这一瞬间，水花在空中绽放，形成美丽的弧线。玻璃杯中的牛奶纯白，番茄的鲜红与之形成鲜明对比。背景简洁，突出主体。近景特写，垂直俯视视角，展现细节之美。"

# control gray
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/gray/src_video.mp4" --prompt "镜头缓缓向右平移，身穿淡黄色坎肩长裙的长发女孩面对镜头露出灿烂的漏齿微笑。她的长发随风轻扬，眼神明亮而充满活力。背景是秋天红色和黄色的树叶，阳光透过树叶的缝隙洒下斑驳光影，营造出温馨自然的氛围。画面风格清新自然，仿佛夏日午后的一抹清凉。中景人像，强调自然光效和细腻的皮肤质感。"

# control pose
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/pose/src_video.mp4" --prompt "在一个热带的庆祝派对上，一家人围坐在椰子树下的长桌旁。桌上摆满了异国风味的美食。长辈们愉悦地交谈，年轻人兴奋地举杯碰撞，孩子们在沙滩上欢乐奔跑。背景中是湛蓝的海洋和明亮的阳光，营造出轻松的气氛。镜头以动态中景捕捉每个开心的瞬间，温暖的阳光映照着他们幸福的面庞。"

# control scribble
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/scribble/src_video.mp4" --prompt "画面中荧光色彩的无人机从极低空高速掠过超现实主义风格的西安古城墙，尘埃反射着阳光。镜头快速切换至城墙上的砖石特写，阳光温暖地洒落，勾勒出每一块砖块的细腻纹理。整体画质清晰华丽，运镜流畅如水。"

# reference face
python inference/vace_wan_inference.py --src_ref_images "benchmarks/VACE-Benchmark/assets/examples/face/src_ref_image_1.png" --prompt "视频展示了一位长着尖耳朵的老人，他有一头银白色的长发和小胡子，穿着一件色彩斑斓的长袍，内搭金色衬衫，散发出神秘与智慧的气息。背景为一个华丽宫殿的内部，金碧辉煌。灯光明亮，照亮他脸上的神采奕奕。摄像机旋转动态拍摄，捕捉老人轻松挥手的动作。"

# reference object
python inference/vace_wan_inference.py --src_ref_images "benchmarks/VACE-Benchmark/assets/examples/object/src_ref_image_1.png" --prompt "经典游戏角色马里奥在绿松石色水下世界中，四周环绕着珊瑚和各种各样的热带鱼。马里奥兴奋地向上跳起，摆出经典的欢快姿势，身穿鲜明的蓝色潜水服，红色的潜水面罩上印有“M”标志，脚上是一双潜水靴。背景中，水泡随波逐流，浮现出一个巨大而友好的海星。摄像机从水底向上快速移动，捕捉他跃出水面的瞬间，灯光明亮而流动。该场景融合了动画与幻想元素，令人惊叹。"

# composition reference_anything
python inference/vace_wan_inference.py --src_ref_images "benchmarks/VACE-Benchmark/assets/examples/reference_anything/src_ref_image_1.png,benchmarks/VACE-Benchmark/assets/examples/reference_anything/src_ref_image_2.png" --prompt "一名打扮成超人的男子自信地站着，面对镜头，肩头有一只充满活力的毛绒黄色鸭子。他留着整齐的短发和浅色胡须，鸭子有橙色的喙和脚，它的翅膀稍微展开，脚分开以保持稳定。他的表情严肃而坚定。他穿着标志性的蓝红超人服装，胸前有黄色“S”标志。斗篷在他身后飘逸。背景有行人。相机位于视线水平，捕捉角色的整个上半身。灯光均匀明亮。"

# composition swap_anything
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/swap_anything/src_video.mp4" --src_mask "benchmarks/VACE-Benchmark/assets/examples/swap_anything/src_mask.mp4" --src_ref_images "benchmarks/VACE-Benchmark/assets/examples/swap_anything/src_ref_image_1.png" --prompt "视频展示了一个人在宽阔的草原上骑马。他有淡紫色长发，穿着传统服饰白上衣黑裤子，动画建模画风，看起来像是在进行某种户外活动或者是在进行某种表演。背景是壮观的山脉云的天空，给人一种宁静而广阔的感觉。整个视频的拍摄角度是固定的，重点展示了骑手和他的马。"

# composition expand_anything
python inference/vace_wan_inference.py --src_video "benchmarks/VACE-Benchmark/assets/examples/expand_anything/src_video.mp4" --src_mask "benchmarks/VACE-Benchmark/assets/examples/expand_anything/src_mask.mp4" --src_ref_images "benchmarks/VACE-Benchmark/assets/examples/expand_anything/src_ref_image_1.png" --prompt "古典油画风格，背景是一条河边，画面中央一位成熟优雅的女人，穿着长裙坐在椅子上。她双手从怀里取出打开的红色心形墨镜戴上。固定机位。"
