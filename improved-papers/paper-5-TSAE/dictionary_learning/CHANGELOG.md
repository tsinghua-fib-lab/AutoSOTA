# CHANGELOG



## v0.1.0 (2025-02-12)

### Feature

* feat: pypi packaging and auto-release with semantic release ([`0ff8888`](https://github.com/saprmarks/dictionary_learning/commit/0ff88883e7caac8ebd7ea0d8e07585451d8b7f9f))

### Unknown

* Merge pull request #37 from chanind/pypi-package

feat: pypi packaging and auto-release with semantic release ([`a711efe`](https://github.com/saprmarks/dictionary_learning/commit/a711efe3b60aabc99a35e7279cd35fa8bf4c930a))

* simplify matryoshka loss ([`43421f5`](https://github.com/saprmarks/dictionary_learning/commit/43421f5934a1476cb3f32f0b9e1b5d14b84540a1))

* Use torch.split() instead of direct indexing for 25% speedup ([`505a445`](https://github.com/saprmarks/dictionary_learning/commit/505a4455358f079db9f2b0309cc0922169869965))

* Fix matryoshka spelling ([`aa45bf6`](https://github.com/saprmarks/dictionary_learning/commit/aa45bf6ed9aa981f6a266f333e6d4a8b9d459909))

* Fix incorrect auxk logging name ([`784a62a`](https://github.com/saprmarks/dictionary_learning/commit/784a62a405be4ee8754a76ad4d3e61fd7de06348))

* Add citation ([`77f2690`](https://github.com/saprmarks/dictionary_learning/commit/77f2690abcd56ce19aaf3c1404dcfcfc6cf9381b))

* Make sure to detach reconstruction before calculating aux loss ([`db2b564`](https://github.com/saprmarks/dictionary_learning/commit/db2b5642e2966559a907e4885bf3317ea997a494))

* Merge pull request #36 from saprmarks/aux_loss_fixes

Aux loss fixes, standardize decoder normalization ([`34eefda`](https://github.com/saprmarks/dictionary_learning/commit/34eefdafcbcac784f3761abf5037c5178cbfd866))

* Standardize and fix topk auxk loss implementation ([`0af1971`](https://github.com/saprmarks/dictionary_learning/commit/0af19713feb5b4c35788039245013736bf974383))

* Normalize decoder after optimzer step ([`200ed3b`](https://github.com/saprmarks/dictionary_learning/commit/200ed3bed09c88d336c25a886eee4cb98c1e616e))

* Remove experimental matroyshka temperature ([`6c2fcfc`](https://github.com/saprmarks/dictionary_learning/commit/6c2fcfc2a8108ba720591eb414be6ab16157dc36))

* Make sure x is on the correct dtype for jumprelu when logging ([`c697d0f`](https://github.com/saprmarks/dictionary_learning/commit/c697d0f83984f0f257be2044231c30f2abb15aa1))

* Import trainers from correct relative location for submodule use ([`8363ff7`](https://github.com/saprmarks/dictionary_learning/commit/8363ff779eee04518edaac9d10d97e459f708b66))

* By default, don&#39;t normalize Gated activations during inference ([`52b0c54`](https://github.com/saprmarks/dictionary_learning/commit/52b0c54ba92630cfb2ae007f020ed447d4a5ba9f))

* Also update context manager for matroyshka threshold ([`65e7af8`](https://github.com/saprmarks/dictionary_learning/commit/65e7af80441e5b601114756afc36a4041cec152f))

* Disable autocast for threshold tracking ([`17aa5d5`](https://github.com/saprmarks/dictionary_learning/commit/17aa5d52f818545afe5fbbe3edf1f774cde92f44))

* Add torch autocast to training loop ([`832f4a3`](https://github.com/saprmarks/dictionary_learning/commit/832f4a32428cda68ec418aff9abe7dca66a9f66e))

* Save state dicts to cpu ([`3c5a5cd`](https://github.com/saprmarks/dictionary_learning/commit/3c5a5cdef682cbeb12e23b825f39709f518e2c0a))

* Add an option to pass LR to TopK trainers ([`8316a44`](https://github.com/saprmarks/dictionary_learning/commit/8316a4418dc4acb70ccad9854d3b05df1b817b9d))

* Add April Update Standard Trainer ([`cfb36ff`](https://github.com/saprmarks/dictionary_learning/commit/cfb36fff661fa60f38a2d1b372b6802517c08257))

* Merge pull request #35 from saprmarks/code_cleanup

Consolidate LR Schedulers, Sparsity Schedulers, and constrained optimizers ([`f19db98`](https://github.com/saprmarks/dictionary_learning/commit/f19db98106302ed1d75dc8380160463ff812b1ad))

* Consolidate LR Schedulers, Sparsity Schedulers, and constrained optimizers ([`9751c57`](https://github.com/saprmarks/dictionary_learning/commit/9751c57731a25c04871e8173d16a0e4d902edc19))

* Merge pull request #34 from adamkarvonen/matroyshka

Add Matroyshka, Fix Jump ReLU training, modify initialization ([`92648d4`](https://github.com/saprmarks/dictionary_learning/commit/92648d4e3d28aa397dbc89c43147aa6faf8874b7))

* Add a verbose option during training ([`0ff687b`](https://github.com/saprmarks/dictionary_learning/commit/0ff687bdc12cba66a0233825cb301df28da3a9db))

* Prevent wandb cuda multiprocessing errors ([`370272a`](https://github.com/saprmarks/dictionary_learning/commit/370272a4aac0ad0e59a2982073aa7b08970712b6))

* Log dead features for batch top k SAEs ([`936a69c`](https://github.com/saprmarks/dictionary_learning/commit/936a69c38a74980830f24fc851c40fb93abe8f07))

* Log number of dead features to wandb ([`77da794`](https://github.com/saprmarks/dictionary_learning/commit/77da7945f520f448b0524e476f539b3a44a4ca43))

* Add trainer number to wandb name ([`3b03b92`](https://github.com/saprmarks/dictionary_learning/commit/3b03b92b97d61a95e98b6f187dad97e939f6f977))

* Add notes ([`810dbb8`](https://github.com/saprmarks/dictionary_learning/commit/810dbb8bdce4ac6f1ce371872297b4f7a104e3f6))

* Add option to ignore bos tokens ([`c2fe5b8`](https://github.com/saprmarks/dictionary_learning/commit/c2fe5b89e78ae4a9d41a4809f4d00b8a3fcd0b36))

* Fix jumprelu training ([`ec961ac`](https://github.com/saprmarks/dictionary_learning/commit/ec961acde2244b98b26bcf796c3ec00b721088bb))

* Use kaiming initialization if specified in paper, fix batch_top_k aux_k_alpha ([`8eaa8b2`](https://github.com/saprmarks/dictionary_learning/commit/8eaa8b2407eabd714bbe7d55fd0c15fcb05fba24))

* Format with ruff ([`3e31571`](https://github.com/saprmarks/dictionary_learning/commit/3e31571b20d3e86823540882ec03c87b155d8e3d))

* Add temperature scaling to matroyshka ([`ceabbc5`](https://github.com/saprmarks/dictionary_learning/commit/ceabbc5233dcf28f0f5afd53e0de850d19f34d78))

* norm the correct decoder dimension ([`5383603`](https://github.com/saprmarks/dictionary_learning/commit/53836033b305142fb6d076a52a7679e0642ddb7a))

* Fix loading matroyshkas from_pretrained() ([`764d4ac`](https://github.com/saprmarks/dictionary_learning/commit/764d4ac4450ea6b7d79de52fdec70c7c1e0dfb79))

* Initial matroyshka implementation ([`8ade55b`](https://github.com/saprmarks/dictionary_learning/commit/8ade55b6eb57ed7c7b06a70187ee68e1056bb95b))

* Make sure we step the learning rate scheduler ([`1df47d8`](https://github.com/saprmarks/dictionary_learning/commit/1df47d83d9dea07d2fb905509b635ac6139bcd48))

* Merge pull request #33 from saprmarks/lr_scheduling

Lr scheduling ([`316dbbe`](https://github.com/saprmarks/dictionary_learning/commit/316dbbe9a905bdab91fb2db63bbc61646e7039a6))

* Properly set new parameters in end to end test ([`e00fd64`](https://github.com/saprmarks/dictionary_learning/commit/e00fd643050584f4cfe15ad41e6a01e29e3c0766))

* Standardize learning rate and sparsity schedules ([`a2d6c43`](https://github.com/saprmarks/dictionary_learning/commit/a2d6c43e94ef068821441d47fef8ae7b3215d09e))

* Merge pull request #32 from saprmarks/add_sparsity_warmup

Add sparsity warmup ([`a11670f`](https://github.com/saprmarks/dictionary_learning/commit/a11670fc6b96b1af3fe8a97175218041f2a9791f))

* Add sparsity warmup for trainers with a sparsity penalty ([`911b958`](https://github.com/saprmarks/dictionary_learning/commit/911b95890e20998df92710a01d158f4663d6834b))

* Clean up lr decay ([`e0db40b`](https://github.com/saprmarks/dictionary_learning/commit/e0db40b8fadcdd1e24c1945829ecd4eb57451fa8))

* Track lr decay implementation ([`f0bb66d`](https://github.com/saprmarks/dictionary_learning/commit/f0bb66d1c25bcb7dc8df62d8dbc3bfd47d26b14c))

* Remove leftover variable, update expected results with standard SAE improvements ([`9687bb9`](https://github.com/saprmarks/dictionary_learning/commit/9687bb9858ef05306227309af99cd5c09d91642a))

* Merge pull request #31 from saprmarks/add_demo

Add option to normalize dataset, track thresholds for TopK SAEs, Fix Standard SAE ([`67a7857`](https://github.com/saprmarks/dictionary_learning/commit/67a7857ca63eb9299c340bc8f9804cdd569df1a9))

* Also scale topk thresholds when scaling biases ([`efd76b1`](https://github.com/saprmarks/dictionary_learning/commit/efd76b138f429bb8e5e969e2e45926e886fdd71b))

* Use the correct standard SAE reconstruction loss, initialize W_dec to W_enc.T ([`8b95ec9`](https://github.com/saprmarks/dictionary_learning/commit/8b95ec9b6e9a6d8d6255092e51b7580dccac70d6))

* Add bias scaling to topk saes ([`484ca01`](https://github.com/saprmarks/dictionary_learning/commit/484ca01f405e5791968883123718fd67ee35f299))

* Fix topk bfloat16 dtype error ([`488a154`](https://github.com/saprmarks/dictionary_learning/commit/488a1545922249cdb9ce5a5885c1931a5c21a37f))

* Add option to normalize dataset activations ([`81968f2`](https://github.com/saprmarks/dictionary_learning/commit/81968f2659082996539f08ea3188a5d2ed327696))

* Remove demo script and graphing notebook ([`57f451b`](https://github.com/saprmarks/dictionary_learning/commit/57f451b5635c4677ab47a4172aa588a5bdffdb4e))

* Track thresholds for topk and batchtopk during training ([`b5821fd`](https://github.com/saprmarks/dictionary_learning/commit/b5821fd87e3676e7a9ab6b87d423c03c57a344dd))

* Track threshold for batchtopk, rename for consistency ([`32d198f`](https://github.com/saprmarks/dictionary_learning/commit/32d198f738c61b0c1109f1803c43e01afb977d3e))

* Modularize demo script ([`dcc02f0`](https://github.com/saprmarks/dictionary_learning/commit/dcc02f04e504331011a54ce851a91976daf15170))

* Begin creation of demo script ([`712eb98`](https://github.com/saprmarks/dictionary_learning/commit/712eb98f78d9537aa3ff01a1d9e007361e67c267))

* Fix JumpReLU training and loading ([`552a8c2`](https://github.com/saprmarks/dictionary_learning/commit/552a8c2c12d41b5d520c99bf3534dff5329f0fde))

* Ensure activation buffer has the correct dtype ([`d416eab`](https://github.com/saprmarks/dictionary_learning/commit/d416eab5de1edfe8ea75c972cdf78d9de68642c2))

* Merge pull request #30 from adamkarvonen/add_tests

Add end to end test, upgrade nnsight to support 0.3.0, fix bugs ([`c4eed3c`](https://github.com/saprmarks/dictionary_learning/commit/c4eed3cca27e93f0ad80cd49057cb862d03c86d7))

* Merge pull request #26 from mntss/batchtokp_aux_fix

Fix BatchTopKSAE training ([`2ec1890`](https://github.com/saprmarks/dictionary_learning/commit/2ec18905045109ec0647bc127bacb794312fc2f6))

* Check for is_tuple to support mlp / attn submodules ([`d350415`](https://github.com/saprmarks/dictionary_learning/commit/d350415e119cacb6547703eb9733daf8ef57075b))

* Change save_steps to a list of ints ([`f1b9b80`](https://github.com/saprmarks/dictionary_learning/commit/f1b9b800bc8e2cc308d4d14690df71f854b30fce))

* Add early stopping in forward pass ([`05fe179`](https://github.com/saprmarks/dictionary_learning/commit/05fe179f5b0616310253deaf758c370071f534fa))

* Obtain better test results using multiple batches ([`067bf7b`](https://github.com/saprmarks/dictionary_learning/commit/067bf7b05470f61b9ed4f38b95be55c5ac45fb8f))

* Fix frac_alive calculation, perform evaluation over multiple batches ([`dc30720`](https://github.com/saprmarks/dictionary_learning/commit/dc3072089c24ce1eb8bc40e9f5248c69a92f5174))

* Complete nnsight 0.2 to 0.3 changes ([`807f6ef`](https://github.com/saprmarks/dictionary_learning/commit/807f6ef735872a5cab68773a315f15bc920c3d72))

* Rename input to inputs per nnsight 0.3.0 ([`9ed4af2`](https://github.com/saprmarks/dictionary_learning/commit/9ed4af245a22e095e932d6065d368c58947d9a3d))

* Add a simple end to end test ([`fe54b00`](https://github.com/saprmarks/dictionary_learning/commit/fe54b001cba976ca96d46add8539580268dc5cb6))

* Create LICENSE ([`32fec9c`](https://github.com/saprmarks/dictionary_learning/commit/32fec9c4556b3acaa709d756e8693edde1e74644))

* Fix BatchTopKSAE training ([`4aea538`](https://github.com/saprmarks/dictionary_learning/commit/4aea5388811284f4fd3daa8fb97916073bfe8841))

* dtype for loading SAEs ([`932e10a`](https://github.com/saprmarks/dictionary_learning/commit/932e10a46523347e8c2da70a10bb8e6dd42d17c6))

* Merge pull request #22 from pleask/jumprelu

Implement jumprelu training ([`713f638`](https://github.com/saprmarks/dictionary_learning/commit/713f6389dde35177c83361f90daaba99b5ac3d08))

* Merge branch &#39;main&#39; into jumprelu ([`099dbbf`](https://github.com/saprmarks/dictionary_learning/commit/099dbbfcdcad07dfc85dd65bfbd15ca9eece70a5))

* Merge pull request #21 from pleask/separate-wandb-runs

Use separate wandb runs for each SAE being trained ([`df60f52`](https://github.com/saprmarks/dictionary_learning/commit/df60f52737f18ce0b1ecd2eb9e08d0706871442d))

* Merge branch &#39;main&#39; into jumprelu ([`3dfc069`](https://github.com/saprmarks/dictionary_learning/commit/3dfc069d39ceeb33ce60581fc7cb17f08ec0e428))

* implement jumprelu training ([`16bdfd9`](https://github.com/saprmarks/dictionary_learning/commit/16bdfd95bc04000b89f81b0496df59f17653a2f8))

* handle no wandb ([`8164d32`](https://github.com/saprmarks/dictionary_learning/commit/8164d32ec79325d3cc31063098b9108386eb15cf))

* Merge pull request #20 from pleask/batchtopk

Implement BatchTopK ([`b001fb0`](https://github.com/saprmarks/dictionary_learning/commit/b001fb0fd358efc7647acf835123a5e874a9a822))

* separate runs for each sae being trained ([`7d3b127`](https://github.com/saprmarks/dictionary_learning/commit/7d3b12778070b88fd39c439751973ac83afbe7a0))

* add batchtopk ([`f08e00b`](https://github.com/saprmarks/dictionary_learning/commit/f08e00b2585ab9a965984af4932614a2e408b6e3))

* Move f_gate to encoder&#39;s dtype ([`43bdb3b`](https://github.com/saprmarks/dictionary_learning/commit/43bdb3b903f7a45ee52b4d865f6d6b7bd60647a3))

* Ensure that x_hat is in correct dtype ([`3376f1b`](https://github.com/saprmarks/dictionary_learning/commit/3376f1bd9d05bedd03179475052d3a26a61fad7a))

* Preallocate buffer memory to lower peak VRAM usage when replenishing buffer ([`90aff63`](https://github.com/saprmarks/dictionary_learning/commit/90aff63b042c50c3c81a3977b62248254115e907))

* Perform logging outside of training loop to lower peak memory usage ([`57f8812`](https://github.com/saprmarks/dictionary_learning/commit/57f8812ff93d4d9ac437d29a74f1d920daa45515))

* Remove triton usage ([`475fece`](https://github.com/saprmarks/dictionary_learning/commit/475feceba9e47d6e74b17c87844253f0a209d75d))

* Revert to triton TopK implementation ([`d94697d`](https://github.com/saprmarks/dictionary_learning/commit/d94697df1783da8b6739e565c3a1bd297b8b1e98))

* Add relative reconstruction bias from GDM Gated SAE paper to evaluate() ([`8984b01`](https://github.com/saprmarks/dictionary_learning/commit/8984b0112e6f9eebcf869aba78ad713b2016d6a6))

* git push origin main:Merge branch &#39;ElanaPearl-small_bug_fixes&#39; into main ([`2d586e4`](https://github.com/saprmarks/dictionary_learning/commit/2d586e417cd30473e1c608146df47eb5767e2527))

* simplifying readme ([`9c46e06`](https://github.com/saprmarks/dictionary_learning/commit/9c46e061eb3b29d055e7221ce92524c6546d2a59))

* simplify readme ([`5c96003`](https://github.com/saprmarks/dictionary_learning/commit/5c9600344e033b5a7834a48914e958b257bcb720))

* add missing imports ([`7f689d9`](https://github.com/saprmarks/dictionary_learning/commit/7f689d9a3a60d577a0d860ac306ae7ba0c71240a))

* fix arg name in trainer_config ([`9577d26`](https://github.com/saprmarks/dictionary_learning/commit/9577d26c92affa71a9dcc3a3b3f6cb905f230388))

* update sae training example code ([`9374546`](https://github.com/saprmarks/dictionary_learning/commit/937454616f087a6e30afa2ae5f6d52ea685ebfee))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`7d405f7`](https://github.com/saprmarks/dictionary_learning/commit/7d405f7d7555444c66121bc853ab027f49c408b0))

* GatedSAE: moved feature re-normalization into encode ([`f628c0e`](https://github.com/saprmarks/dictionary_learning/commit/f628c0ef2ec53d20ffd4d3d06f84100054c358e1))

* documenting JumpReLU SAE support ([`322b6c0`](https://github.com/saprmarks/dictionary_learning/commit/322b6c0c75767b7fe110d1454b9dcd4106bb942b))

* support for JumpReluAutoEncoders ([`57df4e7`](https://github.com/saprmarks/dictionary_learning/commit/57df4e75cbf181e3662058a6609ab2bb5921c9c4))

* Add submodule_name to PAnnealTrainer ([`ecdac03`](https://github.com/saprmarks/dictionary_learning/commit/ecdac0376285912d9468695c024b39100c663b07))

* host SAEs on huggingface ([`0ae37fe`](https://github.com/saprmarks/dictionary_learning/commit/0ae37feeb8beac0fce5036c6ff4188c86627775e))

* fixed batch loading in examine_dimension ([`82485d7`](https://github.com/saprmarks/dictionary_learning/commit/82485d78bcb6d3bcec67965743fac32e6d29ff37))

* Merge pull request #17 from saprmarks/collab

Merge Collab Branch ([`cdf8222`](https://github.com/saprmarks/dictionary_learning/commit/cdf82227d24295fe8a83fbcfe785e6d6d4f2b997))

* moved experimental trainers to collab-dev ([`8d1d581`](https://github.com/saprmarks/dictionary_learning/commit/8d1d581f3df482c77ca99d0839f1677b19ca1ae7))

* Merge branch &#39;main&#39; into collab ([`dda38b9`](https://github.com/saprmarks/dictionary_learning/commit/dda38b94a491261fd92bf9754f1c673221d7f270))

* Update README.md ([`4d6c6a6`](https://github.com/saprmarks/dictionary_learning/commit/4d6c6a6cb5816571e045f3c42c9f5b508d395d83))

* remove a sentence ([`2d40ed5`](https://github.com/saprmarks/dictionary_learning/commit/2d40ed598074c57904e9566d82bbd8ce27b661b5))

* add a list of trainers to the README ([`746927a`](https://github.com/saprmarks/dictionary_learning/commit/746927ae0b597e1fcb69aed58a5e9d4b6103732c))

* add architecture details to README ([`60422a8`](https://github.com/saprmarks/dictionary_learning/commit/60422a87231439425b9e27384352b03bc245365a))

* make wandb integration optional ([`a26c4e5`](https://github.com/saprmarks/dictionary_learning/commit/a26c4e57985458735bbf887685b679d16008de98))

* make wandb integration optional ([`0bdc871`](https://github.com/saprmarks/dictionary_learning/commit/0bdc871a95dae4de17b5116eda38f20d2375ebd1))

* Fix tutorial 404 ([`deb3df7`](https://github.com/saprmarks/dictionary_learning/commit/deb3df7906c8a0d00a4286f42cb65ae27667b2a7))

* Add missing values to config ([`9e44ea9`](https://github.com/saprmarks/dictionary_learning/commit/9e44ea9dc015c6bf919bd61aa40892be1da66dc3))

* changed TrainerTopK class name ([`c52ff00`](https://github.com/saprmarks/dictionary_learning/commit/c52ff008869a021b5f58d1beb80f8afe014757c5))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`c04ee3b`](https://github.com/saprmarks/dictionary_learning/commit/c04ee3b006ae72e69266d0ac2163035aee326b6a))

* fixed loss_recovered to incorporate top_k ([`6be5635`](https://github.com/saprmarks/dictionary_learning/commit/6be563540801caf185069051985b453dacc421d8))

* fixed TopK loss (spotted by Anish) ([`a3b71f7`](https://github.com/saprmarks/dictionary_learning/commit/a3b71f71212b839c8814ffa4223a5026837738c3))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`40bcdf6`](https://github.com/saprmarks/dictionary_learning/commit/40bcdf65b646f0e387d030b0c2211eaf07636b4c))

* naming conventions ([`5ff7fa1`](https://github.com/saprmarks/dictionary_learning/commit/5ff7fa101da07dfdb0663a484214b75c79e02fe0))

* small fix to triton kernel ([`5d21265`](https://github.com/saprmarks/dictionary_learning/commit/5d21265bd390d35b937d10d83cdf617151212cb3))

* small updates for eval ([`585e820`](https://github.com/saprmarks/dictionary_learning/commit/585e82070620771ee5bef4278d4d500b02983e0c))

* added some housekeeping stuff to top_k ([`5559c2c`](https://github.com/saprmarks/dictionary_learning/commit/5559c2c02d84df49531a631d3f4b29ef8acf94c4))

* add support for Top-k SAEs ([`2d549d0`](https://github.com/saprmarks/dictionary_learning/commit/2d549d0d98e400fedf4d7c4127d540f97240b89e))

* add transcoder eval ([`8446f4f`](https://github.com/saprmarks/dictionary_learning/commit/8446f4fc1aa9e7a08ece6e2fd59e6fa9583a7501))

* add transcoder support ([`c590a25`](https://github.com/saprmarks/dictionary_learning/commit/c590a254990691947b244e09849db7b288ed6bee))

* added wandb finish to trainer ([`113c042`](https://github.com/saprmarks/dictionary_learning/commit/113c042101b6df6de60b04c7e65116c3a9460904))

* fixed anneal end bug ([`fbd9ee4`](https://github.com/saprmarks/dictionary_learning/commit/fbd9ee41ed23d65cbaedb43447b64ae4117dab9a))

* added layer and lm_name ([`d173235`](https://github.com/saprmarks/dictionary_learning/commit/d17323572d23067bbba949732a842b1c2c149188))

* adding layer and lm_name to trainer config ([`6168ee0`](https://github.com/saprmarks/dictionary_learning/commit/6168ee0210308a42f3536f5bff19db70e91311ae))

* make tracer_args optional ([`31b2828`](https://github.com/saprmarks/dictionary_learning/commit/31b2828869bd560ac29eafbd3abf06f752063047))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`87d2b58`](https://github.com/saprmarks/dictionary_learning/commit/87d2b58da4b5714a44e6d301f2b5595e6bdd4296))

* bug fix evaluating CE loss with NNsight models ([`f8d81a1`](https://github.com/saprmarks/dictionary_learning/commit/f8d81a1d56b96f34c26fcc9f3feac0cb11ab3065))

* Combining P Annealing and Anthropic Update ([`44318e9`](https://github.com/saprmarks/dictionary_learning/commit/44318e999d6d123daad63fa399935ba339421070))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`43e9ca6`](https://github.com/saprmarks/dictionary_learning/commit/43e9ca63664dafd9a9f23f81b0bf57917a9f36ba))

* removing normalization ([`7a98d77`](https://github.com/saprmarks/dictionary_learning/commit/7a98d77318b3abcc0aab237de455eb33e20f691e))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`5f2b598`](https://github.com/saprmarks/dictionary_learning/commit/5f2b598cdbeb311f32c4ce1e2e816769240bb75e))

* added buffer for NNsight models (not LanguageModel classes) as an extra class. We&#39;ll want to combine the three buffers wo currently have at some point ([`f19d284`](https://github.com/saprmarks/dictionary_learning/commit/f19d2843f9fc64192ddac12f345a4ad910b96310))

* fixed nnsight issues model tracing for chess-gpt ([`7e8c9f9`](https://github.com/saprmarks/dictionary_learning/commit/7e8c9f95cd25bb6bc56def8210852841a30f22fd))

* added W_O projection to HeadBuffer ([`47bd4cd`](https://github.com/saprmarks/dictionary_learning/commit/47bd4cdea4a64563d2f8ba9ab39b246caf9f3c8c))

* added support for training SAEs on individual heads ([`a0e3119`](https://github.com/saprmarks/dictionary_learning/commit/a0e31199f2a02c86328bc4551f1d9a0b89d0d87b))

* added support for training SAEs on individual heads ([`47351b4`](https://github.com/saprmarks/dictionary_learning/commit/47351b4f6ca0bbd42981f73a784e1a395a941025))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`7de0bd3`](https://github.com/saprmarks/dictionary_learning/commit/7de0bd3d062693d8a35f309a6bc8b494c98408a3))

* default hyperparameter adjustments ([`a09346b`](https://github.com/saprmarks/dictionary_learning/commit/a09346b928a9782f57ec137b95d9e7636eda2abf))

* normalization in gated_new ([`104aba2`](https://github.com/saprmarks/dictionary_learning/commit/104aba291b0c17a5ec9e86655a281f457ce14cbc))

* fixing bug where inputs can get overwritten ([`93fd46e`](https://github.com/saprmarks/dictionary_learning/commit/93fd46e3884daf2fb2e17d952b7a4030b0129957))

* fixing tuple bug ([`b05dcaf`](https://github.com/saprmarks/dictionary_learning/commit/b05dcafc816370be8f0584700fd5a882be4a2e8f))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`73b5663`](https://github.com/saprmarks/dictionary_learning/commit/73b5663ed47c91c1c0d2fa8d47c029686fcf8a48))

* multiple steps debugging ([`de3eef1`](https://github.com/saprmarks/dictionary_learning/commit/de3eef10d322502f150dc63e9f71d84c9b777b71))

* adding gradient pursuit function ([`72941f1`](https://github.com/saprmarks/dictionary_learning/commit/72941f10e794401b2a6b682aa097f4db3f7aa1fe))

* bugfix ([`53aabc0`](https://github.com/saprmarks/dictionary_learning/commit/53aabc0ae45464fd3d1d1d384969fe7066d94a7a))

* bugfix ([`91691b5`](https://github.com/saprmarks/dictionary_learning/commit/91691b5b8da50f6b1d44eae501529d72b935752e))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`9ce7d80`](https://github.com/saprmarks/dictionary_learning/commit/9ce7d80ec96e7324c095ffef81039d0e6a896feb))

* logging more things ([`8498a75`](https://github.com/saprmarks/dictionary_learning/commit/8498a754acacca467494182dbf7444b34e1184c3))

* changing initialization for AutoEncoderNew ([`c7ee7ec`](https://github.com/saprmarks/dictionary_learning/commit/c7ee7ec8e7c4bc235cf969f7653a2d99f9bd5723))

* fixing gated SAE encoder scheme ([`4084bc3`](https://github.com/saprmarks/dictionary_learning/commit/4084bc3fa50f0764864630b2fe476722a9303b47))

* changes to gatedSAE API ([`9e001d1`](https://github.com/saprmarks/dictionary_learning/commit/9e001d170c752c1887c27942bf3d6336322a0ff0))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`05b397b`](https://github.com/saprmarks/dictionary_learning/commit/05b397bcc60f3026da7d55aefebfd3b2223273a6))

* changing initialization ([`ebe0d57`](https://github.com/saprmarks/dictionary_learning/commit/ebe0d57c62ebde85386fd7ec59157758e85d3ce3))

* finished combining gated and p-annealing ([`4c08614`](https://github.com/saprmarks/dictionary_learning/commit/4c08614403d51328c672983cb28aaaee846092bc))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`8e0a6f9`](https://github.com/saprmarks/dictionary_learning/commit/8e0a6f998ded264270c019ee9b14ffb9c31d650a))

* gated_anneal first steps ([`ba8b8fa`](https://github.com/saprmarks/dictionary_learning/commit/ba8b8fa1efda86ea843b0a837f98f106ab089448))

* jump SAE ([`873b764`](https://github.com/saprmarks/dictionary_learning/commit/873b764b5a17bdc8704a4a871362e1b03de3ef5f))

* adapted loss logging in p_anneal ([`33997c0`](https://github.com/saprmarks/dictionary_learning/commit/33997c05699862896191dc683c9622efc3e97f95))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`1eecbda`](https://github.com/saprmarks/dictionary_learning/commit/1eecbdaf651dffa1ed4962d79d3f2577d1979e91))

* merging gated and Anthropic SAEs ([`b6a24d0`](https://github.com/saprmarks/dictionary_learning/commit/b6a24d001234e38c2f6b4c52215d65fdcb50a09e))

* revert trainer naming ([`c0af6d9`](https://github.com/saprmarks/dictionary_learning/commit/c0af6d9c20fda36ee700e2884611fd12edc3fb59))

* restored trainer naming ([`2ec3c67`](https://github.com/saprmarks/dictionary_learning/commit/2ec3c6768d21b019ba12d0065876011e85bc2aae))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`fe7e93b`](https://github.com/saprmarks/dictionary_learning/commit/fe7e93bf606a6c8e2e2d13335565455359905345))

* various changes ([`32027ae`](https://github.com/saprmarks/dictionary_learning/commit/32027ae3781e367affc23c3fde5fc504ef49ebc4))

* debug panneal ([`463907d`](https://github.com/saprmarks/dictionary_learning/commit/463907dab4ee91254d0ae674752d3c3803a8044d))

* debug panneal ([`8c00100`](https://github.com/saprmarks/dictionary_learning/commit/8c00100423223dc21d57ee4114f9ef6b38ee209e))

* debug panneal ([`dc632cd`](https://github.com/saprmarks/dictionary_learning/commit/dc632cd69df0c1719ebed7bdd677d7373f37dc74))

* debug panneal ([`166f6a9`](https://github.com/saprmarks/dictionary_learning/commit/166f6a9e582d45728d1e8291c6ab451dbb7a35fd))

* debug panneal ([`bcebaa6`](https://github.com/saprmarks/dictionary_learning/commit/bcebaa6b2adedaecd0779d6551929b3a213aef1e))

* debug pannealing ([`446c568`](https://github.com/saprmarks/dictionary_learning/commit/446c568d32ff7c93c9688c193dd459abe9086ed5))

* p_annealing loss buffer ([`e4d4a35`](https://github.com/saprmarks/dictionary_learning/commit/e4d4a3532536d9b450f39c034c0aabd8e95560fa))

* implement Ben&#39;s p-annealing strategy ([`06a27f0`](https://github.com/saprmarks/dictionary_learning/commit/06a27f096c0e62df695d60d9e1ec7df77c305498))

* panneal changes ([`fe4ff6f`](https://github.com/saprmarks/dictionary_learning/commit/fe4ff6fa5d0c85942b45fffa0bb2908f4d13a2aa))

* logging trainer names to wandb ([`f9c5e45`](https://github.com/saprmarks/dictionary_learning/commit/f9c5e45a85ed345fdd95502a4de7a873c25f8456))

* bugfixes for StandardTrainerNew ([`70acd85`](https://github.com/saprmarks/dictionary_learning/commit/70acd8572b1b5250ac38cb9be04069cb1a6f981e))

* trainer for new anthropic infrastructure ([`531c285`](https://github.com/saprmarks/dictionary_learning/commit/531c28596cbbe296c45a2a6e22ea175e8633f2a1))

* adding r_mag parameter to GSAE ([`198ddf4`](https://github.com/saprmarks/dictionary_learning/commit/198ddf4bd4210b95a11b0a29862fe615d1774fe0))

* gatedSAE trainer ([`3567d6d`](https://github.com/saprmarks/dictionary_learning/commit/3567d6d2a2cb6d810df32b838029daacc354aaaa))

* cosmetic change ([`0200976`](https://github.com/saprmarks/dictionary_learning/commit/0200976ba04409d477e5321b586b844dd545b976))

* GatedAutoEncoder class ([`2cfc47b`](https://github.com/saprmarks/dictionary_learning/commit/2cfc47b42e89c294e14c98969a993f5910604211))

* p annealing not affected by resampling ([`ad8d837`](https://github.com/saprmarks/dictionary_learning/commit/ad8d8371411067c6a031d87faf08f4ec2fe96032))

* integrated trainer update ([`c7613d3`](https://github.com/saprmarks/dictionary_learning/commit/c7613d386a5677451f6da6f9260ceb9d28a3a4d4))

* Merge branch &#39;collab&#39; into p_annealing ([`933b80c`](https://github.com/saprmarks/dictionary_learning/commit/933b80c91a3e49e2e7a761422c629588774370eb))

* fixed p calculation ([`9837a6f`](https://github.com/saprmarks/dictionary_learning/commit/9837a6fa4e88303b7694aa3556485661fa512f1c))

* getting rid of useless seed arguement ([`377c762`](https://github.com/saprmarks/dictionary_learning/commit/377c762d9a9333aed42ad097d393796fcf8a7e57))

* trainer initializes SAE ([`7dffb66`](https://github.com/saprmarks/dictionary_learning/commit/7dffb663a0dcc5f5e3c2855e24e9f8b322704bcc))

* trainer initialized SAE ([`6e80590`](https://github.com/saprmarks/dictionary_learning/commit/6e80590fb441c53df70345bfd20da4fbad7c9cf9))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`c58d23d`](https://github.com/saprmarks/dictionary_learning/commit/c58d23d5a6e2d38c0ff47e42b157f1686f7e98a6))

* changes to lista p_anneal trainers ([`3cc6642`](https://github.com/saprmarks/dictionary_learning/commit/3cc6642b414608e5d0e86c733b0855f927afa52c))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`9dfd3db`](https://github.com/saprmarks/dictionary_learning/commit/9dfd3dbf42d3ad35b0bb32f9d8374ac00201edda))

* decoupled lr warmup and p warmup in p_anneal trainer ([`c3c1645`](https://github.com/saprmarks/dictionary_learning/commit/c3c164540476d69ff4c3bfa7f9a1a4532c4603c0))

* Merge pull request #14 from saprmarks/p_annealing

added annealing and trainer_param_callback ([`61927bc`](https://github.com/saprmarks/dictionary_learning/commit/61927bcf99537a15651a9829a6a261cffad9e65f))

* cosmetic changes to interp ([`4a7966f`](https://github.com/saprmarks/dictionary_learning/commit/4a7966f979ea4b660613c980cdefd48494511955))

* Merge branch &#39;collab&#39; of https://github.com/saprmarks/dictionary_learning into collab ([`c76818e`](https://github.com/saprmarks/dictionary_learning/commit/c76818e4dbf7e980251a6f652529e50cd1b1b7de))

* Merge pull request #13 from jannik-brinkmann/collab

add ListaTrainer ([`d4d2fd9`](https://github.com/saprmarks/dictionary_learning/commit/d4d2fd9b57a4ab380a56b1b5fa0faf1d91a29989))

* additional evluation metrics ([`fa2ec08`](https://github.com/saprmarks/dictionary_learning/commit/fa2ec081e2ff42377eb98b031320933806b2faf7))

* add GroupSAETrainer ([`60e6068`](https://github.com/saprmarks/dictionary_learning/commit/60e6068924a42b8252d11b398b9972205b46ece4))

* added annealing and trainer_param_callback ([`18e3fca`](https://github.com/saprmarks/dictionary_learning/commit/18e3fcaaf5428e998d26a0be80f1be56ffea7981))

* Merge remote-tracking branch &#39;upstream/collab&#39; into collab ([`4650c2a`](https://github.com/saprmarks/dictionary_learning/commit/4650c2a7db87c7ca32db043cb15db8a28450a013))

* fixing neuron resampling ([`a346be9`](https://github.com/saprmarks/dictionary_learning/commit/a346be9abc6644fd59ae493e44ef8fdbd1e339e2))

* improvements to saving and logging ([`4a1d7ae`](https://github.com/saprmarks/dictionary_learning/commit/4a1d7ae76d59713fe0c4722e821ad3882c0aa757))

* can export buffer config ([`d19d8d9`](https://github.com/saprmarks/dictionary_learning/commit/d19d8d956da3e04ab899b93fc67c63b0a7bd5020))

* fixing evaluation.py ([`c91a581`](https://github.com/saprmarks/dictionary_learning/commit/c91a5815e4e11197a8031d21193381f9b596b95c))

* fixing bug in neuron resampling ([`67a03c7`](https://github.com/saprmarks/dictionary_learning/commit/67a03c763feec3bcebd9070389b8481257bdf10b))

* add ListaTrainer ([`880f570`](https://github.com/saprmarks/dictionary_learning/commit/880f5706a42c337e021530855166089b6722e1df))

* fixing neuron resampling in standard trainer ([`3406262`](https://github.com/saprmarks/dictionary_learning/commit/3406262b31dd97f29130532d694aecd62f092f80))

* improvements to training and evaluating ([`b111d40`](https://github.com/saprmarks/dictionary_learning/commit/b111d40898d97123722cda60084f46d0766cd3e2))

* Factoring out SAETrainer class ([`fabd001`](https://github.com/saprmarks/dictionary_learning/commit/fabd001d97f869c01e67ea26f2e02822eba9ab82))

* updating syntax for buffer ([`035a0f9`](https://github.com/saprmarks/dictionary_learning/commit/035a0f9d4ffa8e7307ae637fb801a78c0ea9eb95))

* updating readme for from_pretrained ([`70e8c2a`](https://github.com/saprmarks/dictionary_learning/commit/70e8c2a13682ef12658f92b459c1bf552cb78180))

* from_pretrained ([`db96abc`](https://github.com/saprmarks/dictionary_learning/commit/db96abc96be7ba975bb09a41c7a81b13c2ea5f3e))

* Change syntax for specifying activation dimensions and batch sizes ([`bdf1f19`](https://github.com/saprmarks/dictionary_learning/commit/bdf1f19b292b152b3c4601fc7a77fc6d66cd04c0))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`86c7475`](https://github.com/saprmarks/dictionary_learning/commit/86c7475a945c0a70c0a82c914d9733c8d2bcc651))

* activation_dim for IdentityDict is optional ([`be1b68c`](https://github.com/saprmarks/dictionary_learning/commit/be1b68c0df0de955d722f1739f5c115dfbfbf702))

* update umap requirement ([`776b53e`](https://github.com/saprmarks/dictionary_learning/commit/776b53e506a2c720139d056542a3397d883e2c79))

* Merge pull request #10 from adamkarvonen/shell_script_change

Add sae_set_name to local_path for dictionary downloader ([`33b5a6b`](https://github.com/saprmarks/dictionary_learning/commit/33b5a6be4ea3c76aa918178f2dfcd3f7c81e2b97))

* Add sae_set_name to local_path for dictionary downloader ([`d6163be`](https://github.com/saprmarks/dictionary_learning/commit/d6163be200d28653394c2b9adac540c7a27e2659))

* dispatch no longer needed when loading models ([`69c32ca`](https://github.com/saprmarks/dictionary_learning/commit/69c32ca6fcf1c94c4b7fb7ac8b82fe7257123400))

* removed in_and_out option for activation buffer ([`cf6ad1d`](https://github.com/saprmarks/dictionary_learning/commit/cf6ad1d72de9fc11acba34e73a03799e2b893692))

* updating readme with 10_32768 dictionaries ([`614883f`](https://github.com/saprmarks/dictionary_learning/commit/614883f9476613e7c1c48b951cd3947451e1f534))

* upgrade to nnsight 0.2 ([`cbc5f79`](https://github.com/saprmarks/dictionary_learning/commit/cbc5f7991c9233579c36b4972c6273f3f250f0ef))

* downloader script ([`7a305c5`](https://github.com/saprmarks/dictionary_learning/commit/7a305c583dbbf06f3dbb223387dc3536a489b0de))

* fixing device issue in buffer ([`b1b44f1`](https://github.com/saprmarks/dictionary_learning/commit/b1b44f12e176e73544d863d1d41009a284bc1db5))

* added pretrained_dictionary_downloader.sh ([`0028ebe`](https://github.com/saprmarks/dictionary_learning/commit/0028ebe739ac90e2587a86b92b0aa4b2c0b8497e))

* added pretrained_dictionary_downloader.sh ([`8b63d8d`](https://github.com/saprmarks/dictionary_learning/commit/8b63d8d6d74f51c00b191519d383de7f6052df0b))

* added pretrained_dictionary_downloader.sh ([`6771aff`](https://github.com/saprmarks/dictionary_learning/commit/6771aff6543b320e14fb3db99e0c6fd2613cc905))

* efficiency improvements ([`94844d4`](https://github.com/saprmarks/dictionary_learning/commit/94844d4fa9ce4a593faf9b709cf61a45447f84f3))

* adding identity dict ([`76bd32f`](https://github.com/saprmarks/dictionary_learning/commit/76bd32fe87bf3c7f3ce45d13d6fe6a69c81e05b4))

* debugging interp ([`2f75db3`](https://github.com/saprmarks/dictionary_learning/commit/2f75db31233b1296af97c2002194888715355759))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`86812f5`](https://github.com/saprmarks/dictionary_learning/commit/86812f5dae6a4ebc1605f3b067c27d7b8b96001e))

* warns user when evaluating without enough data ([`246c472`](https://github.com/saprmarks/dictionary_learning/commit/246c472d7efb845875c4aa67a8e0dfd417c28f6d))

* cleaning up interp ([`95d7310`](https://github.com/saprmarks/dictionary_learning/commit/95d7310ef39ed2fe7a496d0a63a142fe569bdcf5))

* examine_dimension returns mbottom_tokens and logit stats ([`40137ff`](https://github.com/saprmarks/dictionary_learning/commit/40137ffe47d9c3ee03e9b46f994c5bd98f5b953e))

* continuing merge ([`db693a6`](https://github.com/saprmarks/dictionary_learning/commit/db693a6c4c290bb670f37c0a7e222e25b6b916c6))

* progress on merge ([`949b3a7`](https://github.com/saprmarks/dictionary_learning/commit/949b3a755c1458e7d216cc02dc5bf7d8e8f62a1a))

* changes to buffer.py ([`792546b`](https://github.com/saprmarks/dictionary_learning/commit/792546b35c45fda3e93abcb0f8cc28f70d0e439c))

* fixing some things in buffer.py ([`f58688e`](https://github.com/saprmarks/dictionary_learning/commit/f58688e574f5353f906a470abfbcc386730fdda6))

* updating requirements ([`a54b496`](https://github.com/saprmarks/dictionary_learning/commit/a54b4961a7ac9996566a3c32f4d216968afac7b1))

* updating requirements ([`a1db591`](https://github.com/saprmarks/dictionary_learning/commit/a1db5917be710c046736574a48bc7f0c2ea98506))

* identity dictionary ([`5e1f35e`](https://github.com/saprmarks/dictionary_learning/commit/5e1f35e09abc20c6ee7bc43cfba6231d97121403))

* bug fix for neuron resampling ([`b281b53`](https://github.com/saprmarks/dictionary_learning/commit/b281b538c1de2b5ce220b429dd3ea4be44c5b72f))

* UMAP visualizations ([`81f8e1f`](https://github.com/saprmarks/dictionary_learning/commit/81f8e1f164def236423e53b89da37d50c115fc62))

* better normalization for ghost_loss ([`fc74af7`](https://github.com/saprmarks/dictionary_learning/commit/fc74af75ca2d9d4fdbca6fefb3feb583ef11583d))

* neuron resampling without replacement ([`4565e9a`](https://github.com/saprmarks/dictionary_learning/commit/4565e9a14975a4a2d9c736ba7c5551b6c9685ae2))

* simplifications to interp functions ([`2318666`](https://github.com/saprmarks/dictionary_learning/commit/231866665154d80e933b5d9ab5be5de5a522c398))

* Second nnsight 0.2 pass through ([`3bcebed`](https://github.com/saprmarks/dictionary_learning/commit/3bcebedb801d5654edb3fc7118144953af2366da))

* Conversion to nnsight 0.2 first pass ([`cac410a`](https://github.com/saprmarks/dictionary_learning/commit/cac410a72e52cbd6f359fd69bd6fdb346923a9e1))

* detaching another thing in ghost grads ([`2f212d6`](https://github.com/saprmarks/dictionary_learning/commit/2f212d6cab348d565633f1bdc0d3e305a6e98d42))

* Neuron resampling no longer errors when resampling zero neurons ([`376dd3b`](https://github.com/saprmarks/dictionary_learning/commit/376dd3b51b1433625386ca357c61497a13b6bf6d))

* NNsight v0.2 Updates ([`90bbc76`](https://github.com/saprmarks/dictionary_learning/commit/90bbc762aaf369a138f544f2e1f3a4e7a6b5fc4a))

* cosmetic improvements to buffer.py ([`b2bd5f0`](https://github.com/saprmarks/dictionary_learning/commit/b2bd5f09cc7f657b7121f0659514d81336903bba))

* fix to ghost grads ([`9531fe5`](https://github.com/saprmarks/dictionary_learning/commit/9531fe5f65a23acb32e0f1c96920d67bb1bed15b))

* fixing table formatting ([`0e69c8c`](https://github.com/saprmarks/dictionary_learning/commit/0e69c8cc7c446db0ddf86da984417965714ec7ec))

* Fixing some table formatting ([`75f927f`](https://github.com/saprmarks/dictionary_learning/commit/75f927f4c722db4c05d64d732b1d025ecdc186aa))

* gpt2-small support ([`f82146c`](https://github.com/saprmarks/dictionary_learning/commit/f82146cf586e53407d639ef81f64e1be481a666b))

* fixing bug relevant to UnifiedTransformer support ([`9ec9ce4`](https://github.com/saprmarks/dictionary_learning/commit/9ec9ce494384ab303db26be066b3a8004230a16a))

* Getting rid of histograms ([`31d09d7`](https://github.com/saprmarks/dictionary_learning/commit/31d09d7136d97c553b8f06c1074ef08ea65be879))

* Fixing tables in readme ([`5934011`](https://github.com/saprmarks/dictionary_learning/commit/59340116bb24cbc01cefa76c52641b5b5b46a340))

* Updates to the readme ([`a5ca51e`](https://github.com/saprmarks/dictionary_learning/commit/a5ca51ea13cfcd4bb286d644e3416a9af3b5fc53))

* Fixing ghost grad bugs ([`633d583`](https://github.com/saprmarks/dictionary_learning/commit/633d583ddaa3090039fca3f1f3e8820ded942e76))

* Handling ghost grad case with no dead neurons ([`4f19425`](https://github.com/saprmarks/dictionary_learning/commit/4f19425a4e09ea93bb7ebaad436c2ef227cb420e))

* adding support for buffer on other devices ([`f3cf296`](https://github.com/saprmarks/dictionary_learning/commit/f3cf296fe00bf547412f7d500b8993796e30a8b9))

* support for ghost grads ([`25d2a62`](https://github.com/saprmarks/dictionary_learning/commit/25d2a62fcaa8bc9be048b5e37aa57441e78262b5))

* add an implementation of ghost gradients ([`2e09210`](https://github.com/saprmarks/dictionary_learning/commit/2e09210099d991d45500488dac9654d141815530))

* fixing a bug with warmup, adding utils ([`47bbde1`](https://github.com/saprmarks/dictionary_learning/commit/47bbde13f47010bbebf6ac393ae3cdc59b804e9d))

* remove HF arg from buffer. rename search_utils to interp ([`7276f17`](https://github.com/saprmarks/dictionary_learning/commit/7276f17288286429162432af6a30763fa80f8117))

* typo fix ([`3f6b922`](https://github.com/saprmarks/dictionary_learning/commit/3f6b922c031f9b31652c3998f0ce1e985629c62a))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`278084b`](https://github.com/saprmarks/dictionary_learning/commit/278084b0a54e5804a064358a1fb28bc007e4fae4))

* added utils for converting hf dataset to generator ([`82fff19`](https://github.com/saprmarks/dictionary_learning/commit/82fff1968ae883afec82d14246041df793ffd170))

* add ablated token effects to ; restore support for HF datasets ([`799e2ca`](https://github.com/saprmarks/dictionary_learning/commit/799e2caeb3f4f4f922531cfb3b14dd34d999ae9d))

* merge in function for examining features ([`986bf96`](https://github.com/saprmarks/dictionary_learning/commit/986bf9646e82f35186c74ce88e6c6e4dc1c8470f))

* easier submodule/dictionary feature examination ([`2c8b985`](https://github.com/saprmarks/dictionary_learning/commit/2c8b98567e1908a4279efc342f46bd4bd72ab618))

* Adding lr warmup after every time neurons are resampled ([`429c582`](https://github.com/saprmarks/dictionary_learning/commit/429c582f84be12d6c326b131f926b33d48698c7b))

* fixing issues with EmptyStream exception ([`39ff6e1`](https://github.com/saprmarks/dictionary_learning/commit/39ff6e1cdccb438d335c39c36656657f974f585f))

* Minor changes due to updates in nnsight ([`49bbbac`](https://github.com/saprmarks/dictionary_learning/commit/49bbbac6a653398be8726587c2c634e0fd831f02))

* Revert &#34;restore support for streaming HF datasets&#34;

This reverts commit b43527b9b6b24521f6eba68242dc22c3c68173d8. ([`23ada98`](https://github.com/saprmarks/dictionary_learning/commit/23ada983527a748887b7481e255b8dfdb310a23d))

* restore support for streaming HF datasets ([`b43527b`](https://github.com/saprmarks/dictionary_learning/commit/b43527b9b6b24521f6eba68242dc22c3c68173d8))

* first version of automatic feature labeling ([`c6753f6`](https://github.com/saprmarks/dictionary_learning/commit/c6753f62967503583aae33978b0684d5af0947e5))

* Add feature_effect function to search_utils.py ([`0ada2c6`](https://github.com/saprmarks/dictionary_learning/commit/0ada2c654b2dcc71e14869afc813b3adce445472))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`fab70b1`](https://github.com/saprmarks/dictionary_learning/commit/fab70b1b1a17fbe46fbdc54ea34095457c8cbe64))

* adding sqrt to MSE ([`63b2174`](https://github.com/saprmarks/dictionary_learning/commit/63b217449c651c78da68571bb032563ac73ebd71))

* Merge pull request #1 from cadentj/main

Update README.md ([`fd79bb3`](https://github.com/saprmarks/dictionary_learning/commit/fd79bb34a7cb56bd987ce8a24764a72586999431))

* Update README.md ([`cf5ec24`](https://github.com/saprmarks/dictionary_learning/commit/cf5ec240bcb31db7007dceb7b4362967b044fd01))

* Update README.md ([`55f33f2`](https://github.com/saprmarks/dictionary_learning/commit/55f33f226d94baace938501d741ccfb5e9816a56))

* evaluation.py ([`2edf59e`](https://github.com/saprmarks/dictionary_learning/commit/2edf59ebb2a625e0862cecd5e4d84249589d95b9))

* evaluating dictionaries ([`71e28fb`](https://github.com/saprmarks/dictionary_learning/commit/71e28fbfa2976b099e849c766176252fa8d9fbc2))

* Removing experimental use of sqrt on MSELoss ([`865bbb5`](https://github.com/saprmarks/dictionary_learning/commit/865bbb58fdd1af681a2a435f546f4f6dceaaf930))

* Adding readme, evaluation, cleaning up ([`ddac948`](https://github.com/saprmarks/dictionary_learning/commit/ddac948a7971e526a47a9dae7311a25c0c56a81c))

* some stuff for saving dicts ([`d1f0e21`](https://github.com/saprmarks/dictionary_learning/commit/d1f0e21afc6395ddec71e274bbd3075750f4a76f))

* removing device from buffer ([`398f15c`](https://github.com/saprmarks/dictionary_learning/commit/398f15cb5d44ba81e12dee5299841a983e9f54df))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`7f013c2`](https://github.com/saprmarks/dictionary_learning/commit/7f013c2441620391eabba4f408deaa14140a5239))

* lr schedule + enabling stretched mlp ([`4eaf7e3`](https://github.com/saprmarks/dictionary_learning/commit/4eaf7e35e8c1c461da761a71968d8e9d1ef0c6b3))

* add random feature search ([`e58cc67`](https://github.com/saprmarks/dictionary_learning/commit/e58cc67cb8303b48cf40cb52e586d464f8cb6b48))

* restore HF support and progress bar ([`7e2b6c6`](https://github.com/saprmarks/dictionary_learning/commit/7e2b6c69aa7095680affe58c4251577f96505915))

* Merge branch &#39;main&#39; of https://github.com/saprmarks/dictionary_learning into main ([`d33ef05`](https://github.com/saprmarks/dictionary_learning/commit/d33ef052e7d4175a5042855c04a6f3b60acb07ff))

* more support for saving checkpints ([`0ca258a`](https://github.com/saprmarks/dictionary_learning/commit/0ca258af3775910ce20d4cce541ff4de962bef3d))

* fix unit column bug + add scheduler ([`5a05c8c`](https://github.com/saprmarks/dictionary_learning/commit/5a05c8cd1b29894e8ba77b115727f1511c3334bd))

* fix merge bugs: checkpointing support ([`9c5bbd8`](https://github.com/saprmarks/dictionary_learning/commit/9c5bbd8a3ac82e8611434d7ba95da172a80a44a0))

* Merge: add HF datasets and checkpointing ([`ccf6ed1`](https://github.com/saprmarks/dictionary_learning/commit/ccf6ed1d9fdc7c0df68c879c893f919d8c192b83))

* checkpointing, progress bar, HF dataset support ([`fd8a3ee`](https://github.com/saprmarks/dictionary_learning/commit/fd8a3ee3ee70354191c4d8ecce9d4f8b878d40c6))

* progress bar for training autoencoders ([`0a8064d`](https://github.com/saprmarks/dictionary_learning/commit/0a8064dd7ef93904c4b5b4edb9fc7ddbc1e42af1))

* implementing neuron resampling ([`f9b9d02`](https://github.com/saprmarks/dictionary_learning/commit/f9b9d020cd5c2daf857d44de2c956a6df2cf7cc3))

* lotsa stuff ([`bc09ba4`](https://github.com/saprmarks/dictionary_learning/commit/bc09ba48a701900311d7049dab52549b8239cb15))

* adding __init__.py file for imports ([`3d9fd43`](https://github.com/saprmarks/dictionary_learning/commit/3d9fd43957b8c35e1d6377aa33341f663ae5d289))

* modifying buffer ([`ba9441b`](https://github.com/saprmarks/dictionary_learning/commit/ba9441b444cd56b2a01c341357d3ede11b06e2b6))

* first commit ([`ea89e90`](https://github.com/saprmarks/dictionary_learning/commit/ea89e90e3f737ec8e2a339cfd0b2f1a1082ef850))

* Initial commit ([`741f4d6`](https://github.com/saprmarks/dictionary_learning/commit/741f4d6e1d07e55f6c6df5340cc22b9c7f8d49b7))
