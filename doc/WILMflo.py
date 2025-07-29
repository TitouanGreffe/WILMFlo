#-----------Import-------------------
import sys, os
import numpy as np
import pandas as pd
import time
path = "[path to ODYM-master/]"
#path = "D:\Documents_D\Documents\PhD\AA_Articles\Articles\II_Operationalize_the_model\Code\AA_Others\ODYM_latest\ODYM-master\ODYM-master"
sys.path.insert(0, os.path.join(path, 'odym', 'modules'))  # add ODYM module directory to system path, relative
sys.path.insert(0, os.path.join(os.getcwd(), path, 'odym',
                                'modules'))  # add ODYM module directory to system path, absolute

import dynamic_stock_model as dsm

class WILMFlo_world:

    def __init__(self,in_folder_path, in_file, scenario_demand, scenario_prod,output_folder, path_L_matrices):

        self.in_folder_path = in_folder_path
        self.in_file = in_file
        self.scenario_demand = scenario_demand
        self.scenario_prod = scenario_prod
        self.output_folder = output_folder
        self.path_L_matrices = path_L_matrices

        self.pertinent_technologies = {}

        ### Load all input data to conduct simulations afterwards
        self.elem_mat = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Elem_Mat", index_col=0).fillna(0)
        self.mat_tec = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Mat_Tec", index_col=0).fillna(0)
        self.tec_services = pd.read_excel(self.in_folder_path + self.in_file,
                                          sheet_name="Tec_Service",
                                          index_col=0).fillna(0)


        self.share_tec_services = pd.read_excel(self.in_folder_path + self.in_file,
                                                sheet_name="ShareTec_Service_" + self.scenario_prod,
                                                index_col=0).fillna(0)
        if self.scenario_demand == "STEPS" and self.scenario_prod == "IEA_NZ":
            '''
            This is a special case for the Net Zero scenario which combines the vector of demand of the STEPS scenario
            with the L^{use} of the Decent Living Standards, which is the Leontief inverse of A matrix with only inputs 
            used to produced energy consumed during the use phase of final technologies
            '''

            self.tec_tec_use_phase = pd.read_excel(self.in_folder_path + self.in_file,
                                                   sheet_name="Tec_Tec_use_DLS", index_col=0).fillna(0)
            self.share_tec_tec = pd.read_excel(self.in_folder_path + self.in_file,
                                               sheet_name="Share_Tec_Tec_" + self.scenario_prod,
                                               index_col=0).fillna(0)
            self.demand = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Demand_" + self.scenario_demand,
                                        index_col=0)

        else:
            #self.tec_tec_use_phase = pd.read_excel(self.in_folder_path + self.in_file,
            #                                       sheet_name="Tec_Tec_use_" + self.scenario_demand,
            #                                       index_col=0).fillna(0)
            #self.share_tec_tec = pd.read_excel(self.in_folder_path + self.in_file,
            #                                   sheet_name="Share_Tec_Tec_" + self.scenario_prod,
            #                                   index_col=0).fillna(0)
            self.demand = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Demand_" + self.scenario_demand,
                                        index_col=0)

        self.tec_tec_prod = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Tec_Tec_prod",
                                              index_col=0).fillna(0)


        self.lists = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Lists")
        self.elements = [i for i in list(self.lists.loc[:, "Element"].dropna()) if
                         i not in list(self.lists.loc[:, "No_Tec_Yet"].dropna())]
        self.hosts = [i for i in list(self.lists.loc[:,"Hosts"].dropna())]
        self.byproducts = [i for i in list(self.lists.loc[:,"Byproducts"].dropna())]
        self.pure_byproducts = [i for i in self.byproducts if i not in self.hosts]
        self.other_elements = [i for i in self.elements if i not in self.pure_byproducts]
        self.materials = [i for i in list(self.lists.loc[:, "Material"].dropna()) if
                          i not in list(self.lists.loc[:, "No_Tec_Yet"].dropna())]
        self.all_products = list(self.lists.loc[:, "Technology"].dropna())
        print("All products = ", self.all_products)
        self.intermediary_products = [i for i in list(self.lists.loc[:, "Material_and_Technology"].dropna()) if
                                           i in self.all_products] + [i for i in list(self.lists.loc[:, "Intermediate_technology"].dropna()) if i in self.all_products]
        self.products = [i for i in self.all_products if i not in self.intermediary_products]
        print("Products only = ", self.products)
        self.final_technologies = [i for i in list(self.tec_services.index)[1:] if i in self.all_products]
        print("self.final_technologies = ", self.final_technologies)

        self.services = list(self.lists.loc[:, "Service"].dropna())

        self.ts = 1
        self.list_time = list(range(2024, 2200, self.ts))
        start = 0
        period = 77
        self.time = self.list_time[start:start + period]

        self.av_BtH = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="av_BtH", index_col=0)


        self.capacity = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Capacity", index_col=0).fillna(0)
        #self.initial_stock = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Initial_stock",
        #                                   index_col=0).fillna(0)

        self.prod_rr = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Prod_RR", index_col=0).fillna(0)

        self.fab_rr = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Fab_RR", index_col=0).fillna(0)

        self.coll_sorting_rr = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Collection_sorting_RR",
                                             index_col=0).fillna(0)

        self.remelting_rr = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Remelting_RR",
                                          index_col=0).fillna(0)

        self.in_use_rr = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="In_Use_RR", index_col=0).fillna(
            0)

        #self.tailings_waste_rock = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Tailings_rock", index_col=0).fillna(
        #   0)

        ## Reserves-cost
        self.reserves_cost = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Reserves_cost", index_col=0)

        #self.initial_stock_time = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Initial_stock_time",
        #                                        index_col=0)
        self.outflow_init_stock_time = pd.read_excel(self.in_folder_path + self.in_file,
                                                     sheet_name="Outflow_init_stock_time",
                                                     index_col=0)

    def flatten(self,xss):
        return [x for xs in xss for x in xs]



    def create_variables(self):

        ## Index elements, columns time
        self.f_ext_supply = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.f_ext = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        self.f_ext_level0 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.f_ext_level1 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.f_ext_level2 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.f_ext_level3 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        self.mining_cost = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        self.reserves_level0 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.reserves_level1 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.reserves_level2 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.reserves_level3 = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        self.gap_reserves = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        self.diss_prim_prod = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.diss_tailings = pd.DataFrame(0, index=self.elements, columns=self.time, dtype=float)
        self.diss_waste_rock = pd.DataFrame(0, index=self.elements, columns=self.time, dtype=float)


        self.diss_yearly_total = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)

        ## to store results of surplus and deficit when considering available byproduct-to-host ratios
        self.deficit_extraction_cap = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.surplus_extraction_cap = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)
        self.upper_bound_f_ext = pd.DataFrame(0,index=self.elements, columns=self.time,dtype=float)


        ## Index materials, columns time
        self.f_rec_fab = pd.DataFrame(0,index=self.materials, columns=self.time,dtype=float)
        self.diss_fab = pd.DataFrame(0,index=self.materials, columns=self.time,dtype=float)
        self.diss_in_use = pd.DataFrame(0,index=self.materials, columns=self.time,dtype=float)
        self.diss_coll_sorting = pd.DataFrame(0,index=self.materials, columns=self.time,dtype=float)
        self.diss_remelting = pd.DataFrame(0,index=self.materials, columns=self.time,dtype=float)


        array_index = [self.flatten([[e] * len(self.materials) for e in self.elements]),
                       self.materials * len(self.elements)]
        tuples = list(zip(*array_index))
        self.f_ext_supply_materials = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('element', 'material')),
                                            columns=self.time,dtype=float)
        array_index = [self.flatten([[m] * len(self.products) for m in self.materials]), self.products * len(self.materials)]
        tuples = list(zip(*array_index))
        self.f_fab_materials = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                            columns=self.time,dtype=float)
        self.f_into_use_1 = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                            columns=self.time,dtype=float)
        self.f_into_use_2 = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                         columns=self.time,dtype=float)
        self.f_into_use_tot = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                         columns=self.time,dtype=float)
        self.demand_stock_in_use = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                            columns=self.time,dtype=float)
        self.diss_in_use_products = pd.DataFrame(0,
                                                index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                                columns=self.time, dtype=float)
        self.f_use_sorting = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                         columns=self.time,dtype=float)
        self.f_sorting_rec = pd.DataFrame(0,index=pd.MultiIndex.from_tuples(tuples, names=('material', 'product')),
                                         columns=self.time,dtype=float)


        self.input_tec = pd.DataFrame(0,index=self.products, columns=self.time,dtype=float)
        self.demand_stock_tec = pd.DataFrame(0,index=self.products, columns=self.time,dtype=float)

        self.outflow_services = pd.DataFrame(0,index=self.services, columns=self.time,dtype=float)


    def run_simulation(self):
        col_amount_mat_tec = "Mean amount (tons/Unit Technology)"
        for t in self.time:
            for s in self.services:
                self.outflow_services.loc[s,t] = sum(self.outflow_init_stock_time.loc[f,t]/self.tec_services.loc[f,s] for f in self.pertinent_tec_services[s])
                print("self.outflow_services.loc[s,t] = ",self.outflow_services.loc[s,t])

        for e in self.elements:
            self.reserves_level0.loc[e, self.time[0]] = self.reserves_cost.loc[e,"R_Level_0"]
            self.reserves_level1.loc[e, self.time[0]] = self.reserves_cost.loc[e, "R_Level_1"]
            self.reserves_level2.loc[e, self.time[0]] = self.reserves_cost.loc[e, "R_Level_2"]
            self.reserves_level3.loc[e, self.time[0]] = self.reserves_cost.loc[e, "R_Level_3"]

        for p in self.products:
            self.input_tec.loc[p, self.time[0]] = sum([self.L_prod[self.time[0]].loc[p,f]*
                                               self.always_value_ShareTec_Services(self.share_tec_services, f, s, self.time[0])*
                                               self.tec_services.loc[f,s]*
                                               (self.outflow_services.loc[s,self.time[0]]) for s in self.services for f in self.final_technologies])

        for t in self.time[1:]:
            start = time.time()
            for p in self.products:
                self.demand_stock_tec.loc[p,t] = sum([self.L_prod[t].loc[p, f]*
                                               self.always_value_ShareTec_Services(self.share_tec_services, f, s, t)*
                                               self.tec_services.loc[f,s]*
                                               self.demand.loc[s,t] for s in self.services for f in self.final_technologies])
                if p == "Food provisioning":
                    self.input_tec.loc[p,t] = max(0,sum([self.L_prod[t].loc[p,f]*
                                               self.always_value_ShareTec_Services(self.share_tec_services, f, s, t)*
                                               self.tec_services.loc[f,s]*
                                               self.demand.loc[s,t] for s in self.services for f in self.final_technologies]))
                else:
                    self.input_tec.loc[p,t] = max(0,sum([self.L_prod[t].loc[p,f]*
                                               self.always_value_ShareTec_Services(self.share_tec_services, f, s, t)*
                                               self.tec_services.loc[f,s]*
                                               (self.demand.loc[s,t]-self.demand.loc[s,t-self.ts]+ self.outflow_services.loc[s,t-self.ts]) for s in self.services for f in self.final_technologies]))

            
            end = time.time()
            
            print("Time first loop = ", end-start)
            start = time.time()
            for m in self.materials:
                for p in self.pertinent_technologies[m]:
                    self.f_into_use_1.loc[(m,p),t] = self.always_value_Mat_Tec(self.mat_tec, m, p, t,col_amount_mat_tec)*self.input_tec.loc[p,t]+self.f_use_sorting.loc[(m,p),t-self.ts]
                    print("self.f_into_use_1.loc[(m,p),t] = ",self.f_into_use_1.loc[(m,p),t])
                    self.f_into_use_2.loc[(m, p), t] = self.always_value_Mat_Tec(self.mat_tec, m, p, t,col_amount_mat_tec)*sum(self.demand_stock_tec.loc[f,t]*self.L_use_phase[t].loc[p, f] for f in self.final_technologies)
                    print("self.f_into_use_2.loc[(m,p),t] = ",self.f_into_use_2.loc[(m,p),t])
                    self.f_into_use_tot.loc[(m, p), t] = self.f_into_use_1.loc[(m,p),t]+self.f_into_use_2.loc[(m,p),t]
                    print("self.f_into_use_tot.loc[(m,p),t] = ",self.f_into_use_tot.loc[(m,p),t])
                    self.f_use_sorting.loc[(m,p),t] = sum([self.RF_yearly(t,c,p)*self.f_into_use_tot.loc[(m, p), c] for c in self.time[:self.time.index(t)]])*self.always_value_m_p_rr(self.in_use_rr, m, p, "In_Use")
                    self.f_sorting_rec.loc[(m,p),t] = self.f_use_sorting.loc[(m,p),t]*self.always_value_m_p_rr(self.coll_sorting_rr, m, p, "Collection_sorting")

                    self.f_fab_materials.loc[(m,p),t] = self.f_into_use_tot.loc[(m, p), t]/self.always_value_m_p_rr(self.fab_rr, m, p, "Fab")

                    if p in self.list_tec_short_lifetime:
                        self.diss_in_use_products.loc[(m, p),t] = self.f_into_use_tot.loc[(m, p),t]*(1-self.always_value_m_p_rr(self.in_use_rr, m, p, "In_Use"))
                    else:
                        self.diss_in_use_products.loc[(m,p),t] = sum([self.RF_yearly(t,c,p)*self.f_into_use_tot.loc[(m, p), c] for c in self.time[:self.time.index(t)]])*(1-self.always_value_m_p_rr(self.in_use_rr, m, p, "In_Use"))

                self.diss_fab.loc[m,t] = sum(self.f_fab_materials.loc[(m,p),t]*(1-self.always_value_m_p_rr(self.fab_rr, m, p, "Fab")) for p in self.pertinent_technologies[m])
                self.diss_in_use.loc[m,t] = sum(self.diss_in_use_products.loc[(m,p),t] for p in self.pertinent_technologies[m])
                self.diss_coll_sorting.loc[m,t] = sum(self.f_use_sorting.loc[(m,p),t]*(1-self.always_value_m_p_rr(self.coll_sorting_rr, m, p, "Collection_sorting")) for p in self.pertinent_technologies[m])
                self.f_rec_fab.loc[m,t] = sum(self.f_sorting_rec.loc[(m,p),t]*self.always_value_m_p_rr(self.remelting_rr, m, p, "Remelting") for p in self.pertinent_technologies[m])
                self.diss_remelting.loc[m,t] = sum(self.f_sorting_rec.loc[(m,p),t]*(1-self.always_value_m_p_rr(self.remelting_rr, m, p, "Remelting")) for p in self.pertinent_technologies[m])
            end = time.time()
            print("Time second loop = ", end - start)
            for e in self.elements:
                print("self.pertinent_mat[e] = ",self.pertinent_mat[e])
                for m in self.pertinent_mat[e]:
                    self.f_ext_supply_materials.loc[(e, m), t] = self.always_value_elem_mat(self.elem_mat, e, m)*(sum([self.f_fab_materials.loc[(m,p),t] for p in self.pertinent_technologies[m]])-self.f_rec_fab.loc[m,t])/self.prod_rr.loc[e,"Production_RR"]
                self.f_ext_supply.loc[e,t] = max(0,sum(self.f_ext_supply_materials.loc[(e, m), t] for m in self.materials))

            for e in self.other_elements:
                self.f_ext.loc[e, t] = min(self.always_value_capacity(self.capacity, e, t), self.f_ext_supply.loc[e, t])
                self.diss_prim_prod.loc[e, t] = self.f_ext.loc[e, t] * (1 - self.prod_rr.loc[e, "Production_RR"])
                self.deficit_extraction_cap.loc[e, t] = max(0, self.f_ext_supply.loc[e, t] - self.always_value_capacity(self.capacity, e, t))
                self.diss_yearly_total.loc[e, t] = self.diss_prim_prod.loc[e, t] + sum((self.diss_fab.loc[m, t] +self.diss_in_use.loc[m, t] + self.diss_coll_sorting.loc[m, t] + self.diss_remelting.loc[m, t])*self.always_value_elem_mat(self.elem_mat, e, m) for m in self.materials)
            for b in self.pure_byproducts:
                self.upper_bound_f_ext.loc[b,t] = min(self.always_value_capacity(self.capacity, b, t),sum(self.f_ext.loc[h, t]*self.av_BtH.loc[h,b] for h in self.hosts))
                self.f_ext.loc[b, t] = min(self.upper_bound_f_ext.loc[b,t], self.f_ext_supply.loc[b, t])
                self.diss_prim_prod.loc[b, t] = self.f_ext.loc[b, t] * (1 - self.prod_rr.loc[b, "Production_RR"])
                ## Deficit and surplus evaluated using available byproduct-to-host ratios
                self.deficit_extraction_cap.loc[b,t] = max(0,self.f_ext_supply.loc[b,t]-self.upper_bound_f_ext.loc[b,t])
                self.surplus_extraction_cap.loc[b,t] = max(0,sum(self.f_ext.loc[h, t]*self.av_BtH.loc[h,b] for h in self.hosts)-self.f_ext_supply.loc[b, t])

                ## Sum dissipative flows over the life cycle
                self.diss_yearly_total.loc[b, t] = self.diss_prim_prod.loc[b, t] + sum((self.diss_fab.loc[m, t] +
                                                                                        self.diss_in_use.loc[m, t] +
                                                                                        self.diss_coll_sorting.loc[m, t] +
                                                                                        self.diss_remelting.loc[
                                                                                            m, t]) * self.always_value_elem_mat(self.elem_mat, b, m) for m in self.materials)
            #for e in self.elements:
            #   self.diss_tailings.loc[e,t] = self.diss_prim_prod.loc[e, t]*self.tailings_waste_rock.loc[e,"Tailings"]
            #    self.diss_waste_rock.loc[e,t] = self.diss_prim_prod.loc[e, t]*self.tailings_waste_rock.loc[e,"Waste rock"]

            for e in self.elements:
                if (self.reserves_level0.loc[e, t - self.ts] >= self.f_ext.loc[e, t]):
                    self.f_ext_level0.loc[e, t] = self.f_ext.loc[e, t]
                    self.reserves_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts] - self.f_ext_level0.loc[e, t]
                    self.reserves_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts]
                    self.reserves_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts]
                    self.reserves_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts]
                if (self.f_ext.loc[e, t] >= self.reserves_level0.loc[e, t - self.ts]) and (
                        self.f_ext.loc[e, t] < (self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts])):
                    self.f_ext_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts]
                    self.f_ext_level1.loc[e, t] = self.f_ext.loc[e, t] - self.f_ext_level0.loc[e, t]
                    self.reserves_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts] - self.f_ext_level0.loc[e, t]
                    self.reserves_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts] - self.f_ext_level1.loc[e, t]
                    self.reserves_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts]
                    self.reserves_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts]
                if (self.f_ext.loc[e, t] >= (self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts])) and (
                        self.f_ext.loc[e, t] < (
                        self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts] + self.reserves_level2.loc[
                    e, t - self.ts])):
                    self.f_ext_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts]
                    self.f_ext_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts]
                    self.f_ext_level2.loc[e, t] = self.f_ext.loc[e, t] - (self.f_ext_level0.loc[e, t] + self.f_ext_level1.loc[e, t])
                    self.reserves_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts] - self.f_ext_level0.loc[e, t]
                    self.reserves_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts] - self.f_ext_level1.loc[e, t]
                    self.reserves_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts] - self.f_ext_level2.loc[e, t]
                    self.reserves_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts]
                if (self.f_ext.loc[e, t] >= (
                        self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts] + self.reserves_level2.loc[
                    e, t - self.ts])) and (
                        self.f_ext.loc[e, t] < (
                        self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts] + self.reserves_level2.loc[
                    e, t - self.ts] + self.reserves_level3.loc[e, t - self.ts])):
                    self.f_ext_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts]
                    self.f_ext_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts]
                    self.f_ext_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts]
                    self.f_ext_level3.loc[e, t] = self.f_ext.loc[e, t] - (
                                self.f_ext_level0.loc[e, t] + self.f_ext_level1.loc[e, t] + self.f_ext_level2.loc[e, t])
                    self.reserves_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts] - self.f_ext_level0.loc[e, t]
                    self.reserves_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts] - self.f_ext_level1.loc[e, t]
                    self.reserves_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts] - self.f_ext_level2.loc[e, t]
                    self.reserves_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts] - self.f_ext_level3.loc[e, t]
                if (self.f_ext.loc[e, t] > (
                        self.reserves_level0.loc[e, t - self.ts] + self.reserves_level1.loc[e, t - self.ts] + self.reserves_level2.loc[
                    e, t - self.ts] + self.reserves_level3.loc[e, t - self.ts])):
                    self.f_ext_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts]
                    self.f_ext_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts]
                    self.f_ext_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts]
                    self.f_ext_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts]
                    self.reserves_level0.loc[e, t] = self.reserves_level0.loc[e, t - self.ts] - self.f_ext_level0.loc[e, t]
                    self.reserves_level1.loc[e, t] = self.reserves_level1.loc[e, t - self.ts] - self.f_ext_level1.loc[e, t]
                    self.reserves_level2.loc[e, t] = self.reserves_level2.loc[e, t - self.ts] - self.f_ext_level2.loc[e, t]
                    self.reserves_level3.loc[e, t] = self.reserves_level3.loc[e, t - self.ts] - self.f_ext_level3.loc[e, t]
                    self.gap_reserves.loc[e, t] = self.f_ext.loc[e, t] - (
                                self.f_ext_level0.loc[e, t] + self.f_ext_level1.loc[e, t] + self.f_ext_level2.loc[e, t] +
                                self.f_ext_level3.loc[e, t])

                self.mining_cost.loc[e,t] = (self.reserves_cost.loc[e,"C_Level_0"]*self.f_ext_level0.loc[e, t]+
                                             self.reserves_cost.loc[e,"C_Level_1"]*self.f_ext_level1.loc[e, t]+
                                             self.reserves_cost.loc[e,"C_Level_2"]*self.f_ext_level2.loc[e, t]+
                                             self.reserves_cost.loc[e,"C_Level_3"]*self.f_ext_level3.loc[e, t])

    def export_results(self):
        start = time.time()
        with pd.ExcelWriter(self.output_folder + "Results_"+self.scenario_demand+"_"+self.scenario_prod+"_"+".xlsx") as writer:
            self.mining_cost.to_excel(writer,sheet_name = "mining_cost")
            self.f_ext_supply.to_excel(writer,sheet_name = "f_ext_supply")
            self.f_ext.to_excel(writer,sheet_name = "f_ext")
            self.deficit_extraction_cap.to_excel(writer,sheet_name = "deficit_extraction_cap_av")
            self.surplus_extraction_cap.to_excel(writer,sheet_name = "surplus_extraction_cap_av")
            self.f_ext_supply_materials.to_excel(writer,sheet_name = "f_ext_supply_materials")
            self.f_fab_materials.to_excel(writer,sheet_name = "f_fab_materials")
            self.f_into_use_tot.to_excel(writer,sheet_name = "f_into_use_tot")
            self.f_use_sorting.to_excel(writer,sheet_name = "f_use_sorting")
            self.f_sorting_rec.to_excel(writer,sheet_name = "f_sorting_rec")
            self.f_rec_fab.to_excel(writer,sheet_name = "f_rec_fab")
            self.input_tec.to_excel(writer,sheet_name = "input_tec")
            self.demand_stock_tec.to_excel(writer,sheet_name = "demand_stock_tec")
            self.f_ext_level0.to_excel(writer,sheet_name = "f_ext_level0")
            self.reserves_level0.to_excel(writer,sheet_name = "reserves_level0")
            self.f_ext_level1.to_excel(writer, sheet_name="f_ext_level1")
            self.reserves_level1.to_excel(writer, sheet_name="reserves_level1")
            self.f_ext_level2.to_excel(writer, sheet_name="f_ext_level2")
            self.reserves_level2.to_excel(writer, sheet_name="reserves_level2")
            self.f_ext_level3.to_excel(writer, sheet_name="f_ext_level3")
            self.reserves_level3.to_excel(writer, sheet_name="reserves_level3")
            self.gap_reserves.to_excel(writer, sheet_name="gap_reserves")
            self.diss_prim_prod.to_excel(writer,sheet_name = "diss_prim_prod")
            #self.diss_tailings.to_excel(writer,sheet_name = "diss_tailings")
            #self.diss_waste_rock.to_excel(writer,sheet_name = "diss_waste_rock")
            self.diss_fab.to_excel(writer,sheet_name = "diss_fab")
            self.diss_in_use_products.to_excel(writer,sheet_name = "diss_in_use_products")
            self.diss_in_use.to_excel(writer,sheet_name = "diss_in_use")
            self.diss_coll_sorting.to_excel(writer,sheet_name = "diss_coll_sorting")
            self.diss_remelting.to_excel(writer,sheet_name = "diss_remelting")
            self.diss_yearly_total.to_excel(writer,sheet_name = "diss_yearly_total")
        end = time.time()
        print("Time to export results =", end-start)



    def define_pertinent_tec_services(self):
        self.pertinent_tec_services = {}
        for s in self.services:
            self.pertinent_tec_services[s] = set(f for f in self.final_technologies if self.always_value_tec_services(self.tec_services, f, s) != 0)

    def define_pertinent_mat(self):
        self.pertinent_mat = {}
        for e in self.elements:
            self.pertinent_mat[e] = set(m for m in self.materials if self.always_value_elem_mat(self.elem_mat, e, m) != 0)

    def calc_outflows_rate(self):
        Nt = Nc = len(self.time) + 2
        Lifespan = pd.read_excel(self.in_folder_path + self.in_file, sheet_name="Lifespan", index_col=0)
        # products = ["Buildings","Smartphones","c-Si PV","CIGS PV"]
        Np = len(self.products)

        """Pre-calculate survival tables of products to reduce computation time"""
        ## Inflow-driven of lifetime table:  the probability of a item added to stock in year m (aka cohort m) leaves in in year n.
        ## This value equals pdf(n,m).
        ## and SF_Array_Products: the share of an inflow in year n (age-cohort) still present at the end of year m (after m-n years).

        self.OF_Array_Products = np.zeros((Nt, Nc, Np))  # density functions, by year, age-cohort, and product.
        self.SF_Array_Products = np.zeros((Nt, Nc, Np))

        self.list_tec_short_lifetime = []
        # PDFs are stored externally because recreating them with scipy.stats is slow.
        # Build pdf array from lifetime distribution: Probability of survival.
        for p in range(0, Np):
            # print(products[p])
            param_list = Lifespan.loc[self.products[p], "Parameters"].split(';')
            if Lifespan.loc[self.products[p], "Type"] == "Fixed":  # Define dictinally for fixed lifetime
                self.lt = {'Type': 'Fixed',
                           'Mean': [float(param_list[0])]}
            elif Lifespan.loc[
                self.products[p], "Type"] == "Normal":  # Define dictionary for normally distributed lifetime
                self.lt = {'Type': 'Normal',
                           'Mean': [float(param_list[0])],
                           'StdDev': [float(param_list[1])]}
            elif Lifespan.loc[self.products[
                p], "Type"] == "Weibull":  # Define dictionary for 3-parameter Weibull-distributed lifetime
                self.lt = {'Type': 'Weibull',
                           'Loc': [float(param_list[0])],
                           'Scale': [float(param_list[2])],
                           'Shape': [float(param_list[1])]}
            else:
                print('Unknown uncertainty distribution.')
            DSM_MaTrace_Lifetime = dsm.DynamicStockModel(t=np.arange(0, Nc, 1),
                                                         lt=self.lt)
            self.OF_Array_Products[:, :, p] = DSM_MaTrace_Lifetime.compute_outflow_pdf()
            self.SF_Array_Products[:, :, p] = DSM_MaTrace_Lifetime.compute_sf()

            if float(param_list[0]) < 1:
                self.list_tec_short_lifetime.append(self.products[p])
        print(self.list_tec_short_lifetime)

    def fraction_survival(self,t,c,p):
        pos_t = self.list_time.index(t)
        pos_c = self.list_time.index(c)
        pos_p = self.products.index(p)
        return self.SF_Array_Products[pos_t,pos_c,pos_p]
        
    def fraction_retirement(self,t,c,p):
        pos_t = self.list_time.index(t)
        pos_c = self.list_time.index(c)
        pos_p = self.products.index(p)
        return 1-self.SF_Array_Products[pos_t,pos_c,pos_p]
        
    def RF_yearly(self,t,c,p):
        '''
        Additional fraction of inflow from cohort c retired at year y. sum(RF_yearly(t,c,p) for t in range(c,2122)) = 1
        :param t:
        :param c:
        :param p:
        :return:
        '''
        if t == c-1:
            return (1 - self.fraction_retirement(t , c, p))
        else:
            return (self.fraction_retirement(t+self.ts, c, p) - self.fraction_retirement(t, c, p))



    def define_pertinent_tec(self):
        col_amount_mat_tec = "Mean amount (tons/Unit Technology)"
        t = 2025
        for m in self.materials:
            self.pertinent_technologies[m] = set(p for p in self.products if self.always_value_Mat_Tec(self.mat_tec, m, p, t, col_amount_mat_tec) != 0)



    def read_L_matrix_prod(self):
        self.L_prod = {}
        '''
        self.L_prod is obtained using calc_L_matrix_prod function taking as input variable self.tec_tec_prod which data is stored 
        in tabsheet of self.in_file named "Tec_Tec_use_ + self.scenario_demand"
        
        For P study, Replaced :
        
        for t in self.time:
            if self.scenario_prod == "IEA_NZ" and self.scenario_demand != "DLS":
                self.L_prod[t] = pd.read_excel(
                    self.path_L_matrices + self.scenario_demand + '_' + "STEPS" + '/L_prod/L_' + str(
                        t) + '.xlsx',
                    index_col=0)
            else:
                self.L_prod[t] = pd.read_excel(self.path_L_matrices + self.scenario_demand + '_' + self.scenario_prod + '/L_prod/L_' + str(t) + '.xlsx',
                                                   index_col=0)
        by:
        for t in self.time:
            self.L_prod[t] = pd.read_excel(self.path_L_matrices + '/L_prod/L_' + str(t) + '.xlsx',
                                                   index_col=0)
        
        '''
        for t in self.time:
            if self.scenario_prod == "IEA_NZ" and self.scenario_demand == "STEPS":
                self.L_prod[t] = pd.read_excel(self.path_L_matrices
                                          + "DLS" + '_' + self.scenario_prod + '/L_prod/L_'
                                          + str(t) + '.xlsx', index_col=0)
            else:
                self.L_prod[t] = pd.read_excel('/home/gtitouan/projects/def-cecileb/gtitouan/WILMFlo/L_matrices/' + self.scenario_demand + '_' + self.scenario_prod + '/L_prod/L_' + str(t) + '.xlsx',
                                                   index_col=0)
    def read_L_matrix_with_yearly_use(self):
        self.L_use = {}
        '''
        self.L_use is obtained using calc_L_matrix_use function taking as input variable self.tec_tec_use_phase which data is stored 
        in tabsheet of self.in_file named "Tec_Tec_prod"
        
        For P study, Replaced :
        
        for t in self.time:
            if self.scenario_prod == "IEA_NZ" and self.scenario_demand != "DLS":
                self.L_use[t] = pd.read_excel(self.path_L_matrices+
                                          + "DLS" + '_' + self.scenario_prod + + '/L_use/L_'
                                          + str(t) + '.xlsx', index_col=0)
            else:
                self.L_use[t] = pd.read_excel(self.path_L_matrices+
                                              + self.scenario_demand + '_' + self.scenario_prod + '/L_use/L_'
                                              + str(t) + '.xlsx', index_col=0)
        
        by:
        for t in self.time:
            self.L_use[t] = pd.read_excel(self.path_L_matrices+ + '/L_use/L_'
                                          + str(t) + '.xlsx', index_col=0)
        '''
        for t in self.time:
            if self.scenario_prod == "IEA_NZ" and self.scenario_demand == "STEPS":
                '''
                For the Net Zero scenario, L_use is the same as for the DLS IEA NZ scenario
                '''
                self.L_use[t] = pd.read_excel(self.path_L_matrices
                                          + "DLS" + '_' + self.scenario_prod + '/L_use/L_'
                                          + str(t) + '.xlsx', index_col=0)
            else:
                self.L_use[t] = pd.read_excel(self.path_L_matrices
                                              + self.scenario_demand + '_' + self.scenario_prod + '/L_use/L_'
                                              + str(t) + '.xlsx', index_col=0)

    def get_L_matrix_use(self):
        self.L_use_phase = {}
        for t in self.time:
            '''
            To obtain the Leontief inverse of the technology matrix A which contains solely allocated technology inputs
            (such as installed capacity of c-Si PV) for the energy supplied to final technologies (1 m^2 of residential
            buildings illuminated and heated) i.e. self.L_use_phase, we substract self.L_prod[t] 
            (which is the Leontief inverse of A matrix containing all inputs 
            for building infrastructures EXCEPT inputs related to the use phase)) to self.L_use 
            (which is the Leontief inverse of A matrix containing all inputs 
            (for building infrastructures AND use phase))
            '''
            self.L_use_phase[t] = self.L_use[t] - self.L_prod[t]
            self.L_use_phase[t] = abs(self.L_use_phase[t].mask(self.L_use_phase[t] < 0).fillna(0))
            ## This second line is to avoid any numerical error with negative value after the Leontief inverse


    def always_value_Mat_Tec(self, df, e, p, t, col_am):
        try:
            a = df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == t)].loc[e, col_am]
            return (df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == t)].loc[e, col_am])
        except:
            pass
        try:
            if t > 2050:
                a = df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == 2050)].loc[e, col_am]
                return (df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == 2050)].loc[e, col_am])
        except:
            pass
        try:
            a = df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == "All")].loc[e, col_am]
            return (df[(df.loc[:, "Technology"] == p) & (df.loc[:, "Year"] == "All")].loc[e, col_am])
        except:
            return 0

    def always_value_ShareTec_Services(self, df, p, s, t):
        try:
            a = df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == t)].loc[p, "Share"]
            return (df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == t)].loc[p, "Share"])
        except:
            pass
        try:
            if t > 2050:
                a = df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == 2050)].loc[p, "Share"]
                return (df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == 2050)].loc[p, "Share"])
        except:
            pass
        try:
            a = df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == "All")].loc[p, "Share"]
            return (df[(df.loc[:, "Service"] == s) & (df.loc[:, "Year"] == "All")].loc[p, "Share"])
        except:
            return 0

    def always_value_capacity(self, df, m, t):
        try:
            return (df[(df.loc[:, "constraint"] == "c2_prod") & (df.loc[:, "Year"] == t)].loc[m, "Amount (tons/year)"])
        except:
            pass
        try:
            if t> 2050:
                a = df[(df.loc[:, "constraint"] == "c2_prod") & (df.loc[:, "Year"] == 2050)].loc[m, "Amount (tons/year)"]
                return(df[(df.loc[:, "constraint"] == "c2_prod") & (df.loc[:, "Year"] == 2050)].loc[m, "Amount (tons/year)"])
        except:
            pass
        try:
            return (
                df[(df.loc[:, "constraint"] == "c2_prod") & (df.loc[:, "Year"] == "All")].loc[m, "Amount (tons/year)"])
        except:
            return (1E11)

    def always_value_cons_stock(self, df, p, t):
        try:
            return (df[(df.loc[:, "constraint"] == "c_stock") & (df.loc[:, "Year"] == t)].loc[p, "Amount (tons/year)"])
        except:
            pass
        try:
            return (
                df[(df.loc[:, "constraint"] == "c_stock") & (df.loc[:, "Year"] == "All")].loc[p, "Amount (tons/year)"])
        except:
            return (1E15)

    def always_value_elem_mat(self, df, e, m):
        try:
            return (df[df.loc[:, "Material"] == m].loc[e, "Amount"] / 100)
        except:
            return (0)

    def always_value_tec_services(self, df, p, s):
        try:
            return (df.loc[p, s])
        except:
            return (0)

    def always_value_m_p_rr(self,df,m,p,stage):
        try:
            return (df[df.loc[:, "Technology"] == p].loc[m, stage + "_RR"])
        except:
            return (1)

    def always_value_tec_tec(self, df, p_row, p_col, t):
        try:
            df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == t)].loc[p_row, "Amount"].sum()
            return (df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == t)].loc[p_row, "Amount"].sum())
        except:
            pass
        try:
            if t > 2050:
                df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == 2050)].loc[p_row, "Amount"].sum()
                return (df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == 2050)].loc[p_row, "Amount"].sum())
        except:
            pass
        try:
            df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == "All")].loc[p_row, "Amount"].sum()
            return (df[(df.loc[:, "Technology"] == p_col) & (df.loc[:, "Year"] == "All")].loc[p_row, "Amount"].sum())
        except:
            return (0)

    def always_value_share_tec_tec(self, df, input_tec, tec, t):
        try:
            a = df[(df.loc[:, "Technology"] == tec) & (df.loc[:, "Year"] == t)].loc[input_tec, "Share"]
            return (df[(df.loc[:, "Technology"] == tec) & (df.loc[:, "Year"] == t)].loc[input_tec, "Share"])
        except:
            pass
        try:
            ## As stated in the manuscript, we assume that share of technologies after 2050 are the same as in 2050
            ## This choice being done due to data gaps.
            if t > 2050:
                b = df[(df.loc[:, "Technology"] == tec) & (df.loc[:, "Year"] == 2050)].loc[input_tec, "Share"]
                return (df[(df.loc[:, "Technology"] == tec) & (df.loc[:, "Year"] == 2050)].loc[input_tec, "Share"])
        except:
            pass
        try:
            return (df[(df.loc[:, "Technology"] == tec) & (df.loc[:, "Year"] == "All")].loc[input_tec, "Share"])
        except:
            return (0)

    def L_matrix_df(self, df):
        ## Function defined to calculate the Leontief inverse (I-A)^-1
        I = pd.DataFrame(index=df.index, columns=df.columns, data=np.eye(len(df.index))).astype('float')
        return (pd.DataFrame(np.linalg.inv((I - df).values.astype('float')), df.index, df.columns))

    def BtH_value(self, df, e_i, e_j):
        try:
            if e_i == e_j:
                return 1
            else:
                return df.loc[e_i, e_j]
        except:
            return (0)
    
    
    def calc_L_matrix_prod(self):
        '''
        Calculate the Leontief inverse of the technology with inputs processes to manufacture primary materials
        and recycle secondary materials
        :return:
        '''
        self.L_prod = {}
        A = pd.DataFrame(index=self.all_products, columns=self.all_products).fillna(0)
        for t in self.time:
            for i in A.index:
                for j in A.columns:
                    # print(i,j)
                    A.loc[i, j] = self.always_value_tec_tec(self.tec_tec_prod, i, j, t)*self.always_value_share_tec_tec(self.share_tec_tec,i,j,t)
                    #A.loc[i, j] = self.always_value_tec_tec(self.tec_tec, i, j, t) * self.model.var_share_tec_tec[
                    #    (i, j, t)]
            self.L_prod[t] = self.L_matrix_df(A)
            ## Here we set to 0 all negative values obtained after inversion (calculation residues after inversion)
            self.L_prod[t] = abs(self.L_prod[t].mask(self.L_prod[t] < 0).fillna(0))
            self.L_prod[t].to_excel(
                self.path_L_matrices + '/L_prod/L_' + str(
                    t) + '.xlsx')

    

    def calc_L_matrix_use(self):
        '''
        Calculate the Leontief inverse of the technology with inputs processes to manufacture primary materials
        and yearly energy consumption (per year) of final technologies
        :return:
        '''
        self.L_use = {}
        A = pd.DataFrame(index=self.all_products, columns=self.all_products).fillna(0)
        for t in self.time:
            for i in A.index:
                for j in A.columns:
                    # print(i,j)
                    A.loc[i, j] = self.always_value_tec_tec(self.tec_tec_use_phase, i, j, t)*self.always_value_share_tec_tec(self.share_tec_tec,i,j,t)
                    #A.loc[i, j] = self.always_value_tec_tec(self.tec_tec, i, j, t) * self.model.var_share_tec_tec[
                    #    (i, j, t)]
            self.L_use[t] = self.L_matrix_df(A)
            ## Here we set to 0 all negative values obtained after inversion (calculation residues after inversion)
            self.L_use[t] = abs(self.L_use[t].mask(self.L_use[t] < 0).fillna(0))
            self.L_use[t].to_excel(
                self.path_L_matrices + '/L_use/L_' + str(
                    t) + '.xlsx')

    def solve_scenario(self):
        self.calc_outflows_rate()
        self.read_L_matrix_prod()
        self.read_L_matrix_with_yearly_use()
        self.get_L_matrix_use()
        self.define_pertinent_mat()
        print("self.define_pertinent_mat() done")
        self.define_pertinent_tec()
        print("self.define_pertinent_tec() done")
        self.define_pertinent_tec_services()
        print("self.define_pertinent_tec_services() done")
        self.create_variables()
        self.run_simulation()
        self.export_results()

