## This source code is from RAdam written by Liyuan Liu
##   (https://github.com/LiyuanLucasLiu/RAdam/tree/master)
##
##
##                                 Apache License
##                           Version 2.0, January 2004
##                        http://www.apache.org/licenses/
##
##   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
##
##   1. Definitions.
##
##      "License" shall mean the terms and conditions for use, reproduction,
##      and distribution as defined by Sections 1 through 9 of this document.
##
##      "Licensor" shall mean the copyright owner or entity authorized by
##      the copyright owner that is granting the License.
##
##      "Legal Entity" shall mean the union of the acting entity and all
##      other entities that control, are controlled by, or are under common
##      control with that entity. For the purposes of this definition,
##      "control" means (i) the power, direct or indirect, to cause the
##      direction or management of such entity, whether by contract or
##      otherwise, or (ii) ownership of fifty percent (50%) or more of the
##      outstanding shares, or (iii) beneficial ownership of such entity.
##
##      "You" (or "Your") shall mean an individual or Legal Entity
##      exercising permissions granted by this License.
##
##      "Source" form shall mean the preferred form for making modifications,
##      including but not limited to software source code, documentation
##      source, and configuration files.
##
##      "Object" form shall mean any form resulting from mechanical
##      transformation or translation of a Source form, including but
##      not limited to compiled object code, generated documentation,
##      and conversions to other media types.
##
##      "Work" shall mean the work of authorship, whether in Source or
##      Object form, made available under the License, as indicated by a
##      copyright notice that is included in or attached to the work
##      (an example is provided in the Appendix below).
##
##      "Derivative Works" shall mean any work, whether in Source or Object
##      form, that is based on (or derived from) the Work and for which the
##      editorial revisions, annotations, elaborations, or other modifications
##      represent, as a whole, an original work of authorship. For the purposes
##      of this License, Derivative Works shall not include works that remain
##      separable from, or merely link (or bind by name) to the interfaces of,
##      the Work and Derivative Works thereof.
##
##      "Contribution" shall mean any work of authorship, including
##      the original version of the Work and any modifications or additions
##      to that Work or Derivative Works thereof, that is intentionally
##      submitted to Licensor for inclusion in the Work by the copyright owner
##      or by an individual or Legal Entity authorized to submit on behalf of
##      the copyright owner. For the purposes of this definition, "submitted"
##      means any form of electronic, verbal, or written communication sent
##      to the Licensor or its representatives, including but not limited to
##      communication on electronic mailing lists, source code control systems,
##      and issue tracking systems that are managed by, or on behalf of, the
##      Licensor for the purpose of discussing and improving the Work, but
##      excluding communication that is conspicuously marked or otherwise
##      designated in writing by the copyright owner as "Not a Contribution."
##
##      "Contributor" shall mean Licensor and any individual or Legal Entity
##      on behalf of whom a Contribution has been received by Licensor and
##      subsequently incorporated within the Work.
##
##   2. Grant of Copyright License. Subject to the terms and conditions of
##      this License, each Contributor hereby grants to You a perpetual,
##      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
##      copyright license to reproduce, prepare Derivative Works of,
##      publicly display, publicly perform, sublicense, and distribute the
##      Work and such Derivative Works in Source or Object form.
##
##   3. Grant of Patent License. Subject to the terms and conditions of
##      this License, each Contributor hereby grants to You a perpetual,
##      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
##      (except as stated in this section) patent license to make, have made,
##      use, offer to sell, sell, import, and otherwise transfer the Work,
##      where such license applies only to those patent claims licensable
##      by such Contributor that are necessarily infringed by their
##      Contribution(s) alone or by combination of their Contribution(s)
##      with the Work to which such Contribution(s) was submitted. If You
##      institute patent litigation against any entity (including a
##      cross-claim or counterclaim in a lawsuit) alleging that the Work
##      or a Contribution incorporated within the Work constitutes direct
##      or contributory patent infringement, then any patent licenses
##      granted to You under this License for that Work shall terminate
##      as of the date such litigation is filed.
##
##   4. Redistribution. You may reproduce and distribute copies of the
##      Work or Derivative Works thereof in any medium, with or without
##      modifications, and in Source or Object form, provided that You
##      meet the following conditions:
##
##      (a) You must give any other recipients of the Work or
##          Derivative Works a copy of this License; and
##
##      (b) You must cause any modified files to carry prominent notices
##          stating that You changed the files; and
##
##      (c) You must retain, in the Source form of any Derivative Works
##          that You distribute, all copyright, patent, trademark, and
##          attribution notices from the Source form of the Work,
##          excluding those notices that do not pertain to any part of
##          the Derivative Works; and
##
##      (d) If the Work includes a "NOTICE" text file as part of its
##          distribution, then any Derivative Works that You distribute must
##          include a readable copy of the attribution notices contained
##          within such NOTICE file, excluding those notices that do not
##          pertain to any part of the Derivative Works, in at least one
##          of the following places: within a NOTICE text file distributed
##          as part of the Derivative Works; within the Source form or
##          documentation, if provided along with the Derivative Works; or,
##          within a display generated by the Derivative Works, if and
##          wherever such third-party notices normally appear. The contents
##          of the NOTICE file are for informational purposes only and
##          do not modify the License. You may add Your own attribution
##          notices within Derivative Works that You distribute, alongside
##          or as an addendum to the NOTICE text from the Work, provided
##          that such additional attribution notices cannot be construed
##          as modifying the License.
##
##      You may add Your own copyright statement to Your modifications and
##      may provide additional or different license terms and conditions
##      for use, reproduction, or distribution of Your modifications, or
##      for any such Derivative Works as a whole, provided Your use,
##      reproduction, and distribution of the Work otherwise complies with
##      the conditions stated in this License.
##
##   5. Submission of Contributions. Unless You explicitly state otherwise,
##      any Contribution intentionally submitted for inclusion in the Work
##      by You to the Licensor shall be under the terms and conditions of
##      this License, without any additional terms or conditions.
##      Notwithstanding the above, nothing herein shall supersede or modify
##      the terms of any separate license agreement you may have executed
##      with Licensor regarding such Contributions.
##
##   6. Trademarks. This License does not grant permission to use the trade
##      names, trademarks, service marks, or product names of the Licensor,
##      except as required for reasonable and customary use in describing the
##      origin of the Work and reproducing the content of the NOTICE file.
##
##   7. Disclaimer of Warranty. Unless required by applicable law or
##      agreed to in writing, Licensor provides the Work (and each
##      Contributor provides its Contributions) on an "AS IS" BASIS,
##      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
##      implied, including, without limitation, any warranties or conditions
##      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
##      PARTICULAR PURPOSE. You are solely responsible for determining the
##      appropriateness of using or redistributing the Work and assume any
##      risks associated with Your exercise of permissions under this License.
##
##   8. Limitation of Liability. In no event and under no legal theory,
##      whether in tort (including negligence), contract, or otherwise,
##      unless required by applicable law (such as deliberate and grossly
##      negligent acts) or agreed to in writing, shall any Contributor be
##      liable to You for damages, including any direct, indirect, special,
##      incidental, or consequential damages of any character arising as a
##      result of this License or out of the use or inability to use the
##      Work (including but not limited to damages for loss of goodwill,
##      work stoppage, computer failure or malfunction, or any and all
##      other commercial damages or losses), even if such Contributor
##      has been advised of the possibility of such damages.
##
##   9. Accepting Warranty or Additional Liability. While redistributing
##      the Work or Derivative Works thereof, You may choose to offer,
##      and charge a fee for, acceptance of support, warranty, indemnity,
##      or other liability obligations and/or rights consistent with this
##      License. However, in accepting such obligations, You may act only
##      on Your own behalf and on Your sole responsibility, not on behalf
##      of any other Contributor, and only if You agree to indemnify,
##      defend, and hold each Contributor harmless for any liability
##      incurred by, or claims asserted against, such Contributor by reason
##      of your accepting any such warranty or additional liability.
##
##   END OF TERMS AND CONDITIONS
##
##   APPENDIX: How to apply the Apache License to your work.
##
##      To apply the Apache License to your work, attach the following
##      boilerplate notice, with the fields enclosed by brackets "[]"
##      replaced with your own identifying information. (Don't include
##      the brackets!)  The text should be enclosed in the appropriate
##      comment syntax for the file format. We also recommend that a
##      file or class name and description of purpose be included on the
##      same "printed page" as the copyright notice for easier
##      identification within third-party archives.
##
##   Copyright [2019] [Liyuan Liu]
##
##   Licensed under the Apache License, Version 2.0 (the "License");
##   you may not use this file except in compliance with the License.
##   You may obtain a copy of the License at
##
##       http://www.apache.org/licenses/LICENSE-2.0
##
##   Unless required by applicable law or agreed to in writing, software
##   distributed under the License is distributed on an "AS IS" BASIS,
##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##   See the License for the specific language governing permissions and
##   limitations under the License.

import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class PlainRAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(PlainRAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PlainRAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, warmup = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, warmup = warmup)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['warmup'] > state['step']:
                    scheduled_lr = 1e-8 + state['step'] * group['lr'] / group['warmup']
                else:
                    scheduled_lr = group['lr']

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * scheduled_lr, p_data_fp32)

                p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                p.data.copy_(p_data_fp32)

        return loss
