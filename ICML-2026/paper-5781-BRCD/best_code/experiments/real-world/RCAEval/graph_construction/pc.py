import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

background_knowledge = BackgroundKnowledge()
background_knowledge.add_forbidden_by_pattern(".*mem$", ".*lat50$")
background_knowledge.add_forbidden_by_pattern(".*cpu$", ".*lat50$")
background_knowledge.add_forbidden_by_pattern(".*", "frontend.*")



def pc_default(data, show_progress=False, with_bg=False, **kwargs):
    node_names = data.columns.to_list()

    cg = pc(
        data.to_numpy().astype(float),
        node_names=node_names,
        show_progress=show_progress,
        background_knowledge=background_knowledge if with_bg else None,
    )
    return cg.G.graph


def pc_fisherz(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=fisherz,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_fisherz_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=fisherz,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_gsq(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=gsq,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_gsq_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=gsq,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_chisq(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=chisq,
        stable=False,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pc_chisq_stable(data):
    # data: pd.DataFrame
    node_names = data.columns.to_list()
    data = data.to_numpy()
    cg = pc(
        data=data,
        alpha=0.05,
        indep_test=chisq,
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        background_knowledge=None,
        show_progress=False,
        node_names=node_names,
    )
    return cg


def pcmci(data):
    raise NotImplementedError