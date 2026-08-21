# Third-Party Notices

`src/ClassificationTree.jl` is a modified copy of classification-tree code
from [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl). Changes
made for MHABR include the additional split-evaluation metadata used by the
algorithm.

The bundled `examples/data/iris.csv` file is byte-identical to
`test/data/iris.csv` from DecisionTree.jl v0.12.3. Dataset-specific
attribution and license information is provided in
[`examples/data/README.md`](examples/data/README.md).

DecisionTree.jl states that its code is released under the MIT License and was
originally adapted from MILK: Machine Learning Toolkit:

> Copyright (c) 2008–2011, Luis Pedro Coelho <luis@luispedro.org>
>
> Copyright (c) 2012–2013, Ben Sadeghi

The MIT permission and warranty terms for these portions are reproduced by the
project-level `LICENSE` file. The original copyright notices above must be
retained in redistributions.

DecisionTree.jl also identifies portions of its classification-tree
implementation as a small port from scikit-learn and NumPy, distributed under
the BSD 3-Clause License. Those portions retain the following conditions:

1. Redistributions of source code must retain the copyright notice, this list
   of conditions, and the disclaimer.
2. Redistributions in binary form must reproduce the copyright notice, this
   list of conditions, and the disclaimer in accompanying documentation or
   materials.
3. The copyright holders' and contributors' names may not be used to endorse
   or promote derived products without specific prior written permission.

THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE,
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

See the upstream
[DecisionTree.jl license](https://github.com/JuliaAI/DecisionTree.jl/blob/master/LICENSE.md)
and source headers for the authoritative notices.
